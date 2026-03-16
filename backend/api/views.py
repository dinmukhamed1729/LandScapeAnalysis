import os
import uuid
import threading
import numpy as np
from pathlib import Path
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser

# In-memory job store (use Redis/Celery in production)
JOBS: dict = {}


def compute_index(bands: dict, index_name: str, custom_formula: str = '') -> np.ndarray:
    """Compute vegetation index from band arrays."""
    B = bands.get('B', np.zeros((1, 1)))
    G = bands.get('G', np.zeros((1, 1)))
    R = bands.get('R', np.zeros((1, 1)))
    RE = bands.get('RE', np.zeros((1, 1)))
    NIR = bands.get('NIR', np.zeros((1, 1)))

    eps = 1e-10

    formulas = {
        'NDVI':  (NIR - R) / (NIR + R + eps),
        'NDRE':  (NIR - RE) / (NIR + RE + eps),
        'GNDVI': (NIR - G) / (NIR + G + eps),
        'EVI':   2.5 * (NIR - R) / (NIR + 6 * R - 7.5 * B + 1 + eps),
        'SAVI':  (NIR - R) / (NIR + R + 0.5 + eps) * 1.5,
        'NDWI':  (G - NIR) / (G + NIR + eps),
    }

    if index_name == 'CUSTOM' and custom_formula:
        local_vars = {'B': B, 'G': G, 'R': R, 'RE': RE, 'NIR': NIR, 'np': np, 'eps': eps}
        result = eval(custom_formula, {"__builtins__": {}}, local_vars)  # noqa: S307
        return np.clip(result, -1, 1)

    result = formulas.get(index_name)
    if result is None:
        raise ValueError(f'Unknown index: {index_name}')
    return np.clip(result, -1, 1)


def load_bands_from_file(filepath: str) -> dict:
    """Load band arrays from GeoTIFF. Returns dict of band arrays."""
    try:
        import rasterio
        with rasterio.open(filepath) as src:
            count = src.count
            band_names = ['B', 'G', 'R', 'RE', 'NIR']
            bands = {}
            for i in range(1, min(count + 1, 6)):
                arr = src.read(i).astype(np.float32)
                arr = np.where(arr == src.nodata, np.nan, arr) if src.nodata else arr
                # Normalize to 0-1 if integer dtype
                if src.dtypes[i - 1] in ('uint8', 'uint16'):
                    max_val = np.iinfo(np.dtype(src.dtypes[i - 1])).max
                    arr = arr / max_val
                bands[band_names[i - 1]] = arr
            # If single band, treat as NIR for demo
            if count == 1:
                bands['NIR'] = bands['B']
                bands['R'] = bands['B'] * 0.6
            return bands
    except ImportError:
        # rasterio not available: generate synthetic bands from Pillow
        from PIL import Image
        img = Image.open(filepath).convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0
        return {
            'B': arr[:, :, 2],
            'G': arr[:, :, 1],
            'R': arr[:, :, 0],
            'RE': arr[:, :, 0] * 0.9,
            'NIR': (arr[:, :, 0] * 0.3 + arr[:, :, 1] * 0.6 + arr[:, :, 2] * 0.1),
        }


def index_to_png_base64(index_arr: np.ndarray) -> str:
    """Convert index array to base64 PNG with colormap."""
    import io, base64
    from PIL import Image

    valid = index_arr[~np.isnan(index_arr)]
    vmin, vmax = (float(valid.min()), float(valid.max())) if len(valid) else (-1, 1)
    norm = (index_arr - vmin) / (vmax - vmin + 1e-10)
    norm = np.clip(norm, 0, 1)

    # Green colormap: low=red, mid=yellow, high=green
    r = np.where(norm < 0.5, 1.0, 2.0 - 2.0 * norm)
    g = np.where(norm < 0.5, 2.0 * norm, 1.0)
    b = np.zeros_like(norm)
    rgb = (np.stack([r, g, b], axis=2) * 255).astype(np.uint8)

    img = Image.fromarray(rgb, 'RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def run_analysis(task_id: str, files: list[tuple[str, str]], index_name: str, custom_formula: str):
    """Background analysis worker."""
    try:
        JOBS[task_id]['status'] = 'running'
        results = []

        for filepath, filename in files:
            JOBS[task_id]['progress'] = f'Обработка {filename}...'
            bands = load_bands_from_file(filepath)
            idx_arr = compute_index(bands, index_name, custom_formula)

            valid = idx_arr[~np.isnan(idx_arr)]
            stats = {
                'mean':   round(float(np.nanmean(valid)), 4) if len(valid) else 0,
                'min':    round(float(np.nanmin(valid)), 4) if len(valid) else 0,
                'max':    round(float(np.nanmax(valid)), 4) if len(valid) else 0,
                'median': round(float(np.nanmedian(valid)), 4) if len(valid) else 0,
                'std':    round(float(np.nanstd(valid)), 4) if len(valid) else 0,
            }
            preview_b64 = index_to_png_base64(idx_arr)
            results.append({
                'filename': filename,
                'index': index_name,
                'stats': stats,
                'preview': preview_b64,
            })

        JOBS[task_id].update({'status': 'done', 'progress': 'Готово', 'results': results})
    except Exception as exc:
        JOBS[task_id].update({'status': 'error', 'progress': str(exc)})


class AnalyzeView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        files = request.FILES.getlist('files')
        index_name = request.data.get('index', 'NDVI').upper()
        custom_formula = request.data.get('formula', '')

        if not files:
            return Response({'error': 'Файлы не загружены'}, status=status.HTTP_400_BAD_REQUEST)

        task_id = str(uuid.uuid4())
        JOBS[task_id] = {'status': 'queued', 'progress': 'В очереди...', 'results': []}

        # Save uploaded files
        media_dir = Path(settings.MEDIA_ROOT) / task_id
        media_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for f in files:
            dest = media_dir / f.name
            with open(dest, 'wb') as fh:
                for chunk in f.chunks():
                    fh.write(chunk)
            saved.append((str(dest), f.name))

        thread = threading.Thread(target=run_analysis, args=(task_id, saved, index_name, custom_formula), daemon=True)
        thread.start()

        return Response({'task_id': task_id}, status=status.HTTP_202_ACCEPTED)


class StatusView(APIView):
    def get(self, request, task_id):
        job = JOBS.get(task_id)
        if not job:
            return Response({'error': 'Задача не найдена'}, status=status.HTTP_404_NOT_FOUND)
        return Response(job)
