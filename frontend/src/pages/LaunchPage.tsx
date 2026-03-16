import { useState, useRef, useCallback } from 'react'
import axios from 'axios'
import styles from './LaunchPage.module.css'

const INDICES = ['NDVI', 'NDRE', 'GNDVI', 'EVI', 'SAVI', 'NDWI', 'CUSTOM'] as const

interface PredictResult {
  prediction: number
  input_size: number
  seq_len: number
}
type Index = typeof INDICES[number]

interface Stats {
  mean: number
  min: number
  max: number
  median: number
  std: number
}

interface Result {
  filename: string
  index: string
  stats: Stats
  preview: string
}

interface Job {
  status: 'queued' | 'running' | 'done' | 'error'
  progress: string
  results: Result[]
}

export default function LaunchPage() {
  const [files, setFiles] = useState<File[]>([])
  const [index, setIndex] = useState<Index>('NDVI')
  const [formula, setFormula] = useState('')
  const [dragging, setDragging] = useState(false)
  const [job, setJob] = useState<Job | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const pollRef = useRef<number | null>(null)

  // LSTM prediction state
  const [lstmInput, setLstmInput] = useState('')
  const [lstmLoading, setLstmLoading] = useState(false)
  const [lstmError, setLstmError] = useState('')
  const [lstmResult, setLstmResult] = useState<PredictResult | null>(null)

  const addFiles = (incoming: FileList | null) => {
    if (!incoming) return
    setFiles(prev => {
      const existing = new Set(prev.map(f => f.name))
      const fresh = Array.from(incoming).filter(f => !existing.has(f.name))
      return [...prev, ...fresh]
    })
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    addFiles(e.dataTransfer.files)
  }, [])

  const removeFile = (name: string) =>
    setFiles(prev => prev.filter(f => f.name !== name))

  const poll = (taskId: string) => {
    pollRef.current = window.setInterval(async () => {
      try {
        const { data } = await axios.get<Job>(`/api/status/${taskId}/`)
        setJob(data)
        if (data.status === 'done' || data.status === 'error') {
          clearInterval(pollRef.current!)
          setLoading(false)
        }
      } catch {
        clearInterval(pollRef.current!)
        setLoading(false)
        setError('Ошибка получения статуса задачи')
      }
    }, 1000)
  }

  const handleLaunch = async () => {
    if (!files.length) { setError('Загрузите хотя бы один файл'); return }
    setError('')
    setJob(null)
    setLoading(true)

    const form = new FormData()
    files.forEach(f => form.append('files', f))
    form.append('index', index)
    if (index === 'CUSTOM') form.append('formula', formula)

    try {
      const { data } = await axios.post<{ task_id: string }>('/api/analyze/', form)
      setJob({ status: 'queued', progress: 'В очереди...', results: [] })
      poll(data.task_id)
    } catch {
      setLoading(false)
      setError('Ошибка запуска анализа')
    }
  }

  const handlePredict = async () => {
    setLstmError('')
    setLstmResult(null)

    // Parse input: each line is one timestep; values separated by comma/space
    const lines = lstmInput.trim().split('\n').filter(l => l.trim())
    if (!lines.length) { setLstmError('Введите данные временного ряда'); return }

    const sequence = lines.map(line =>
      line.split(/[\s,;]+/).filter(Boolean).map(Number)
    )
    const hasNaN = sequence.some(row => row.some(isNaN))
    if (hasNaN) { setLstmError('Все значения должны быть числами'); return }

    setLstmLoading(true)
    try {
      const { data } = await axios.post<PredictResult>('/api/predict/', { sequence })
      setLstmResult(data)
    } catch (e: any) {
      setLstmError(e?.response?.data?.error ?? 'Ошибка предсказания')
    } finally {
      setLstmLoading(false)
    }
  }

  const handleReset = () => {
    if (pollRef.current) clearInterval(pollRef.current)
    setFiles([])
    setJob(null)
    setError('')
    setLoading(false)
  }

  const statusColor = job?.status === 'done' ? '#22c55e'
    : job?.status === 'error' ? '#ef4444'
    : '#f59e0b'

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div className={styles.headerInner}>
          <div className={styles.logo}>
            <span className={styles.logoIcon}>🛰</span>
            <div>
              <h1 className={styles.title}>Анализ изменений ландшафта</h1>
              <p className={styles.subtitle}>Вегетационные индексы по мультиспектральным снимкам</p>
            </div>
          </div>
        </div>
      </header>

      <main className={styles.main}>
        {/* Step 1 — Upload */}
        <section className={styles.card}>
          <h2 className={styles.cardTitle}><span className={styles.step}>1</span> Загрузка снимков</h2>
          <div
            className={`${styles.dropzone} ${dragging ? styles.dropzoneActive : ''}`}
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".tif,.tiff,.png,.jpg,.jpeg"
              style={{ display: 'none' }}
              onChange={e => addFiles(e.target.files)}
            />
            <div className={styles.dropIcon}>📁</div>
            <p className={styles.dropText}>Перетащите файлы или кликните для выбора</p>
            <p className={styles.dropHint}>GeoTIFF, DJI-каналы (_B, _G, _R, _RE, _N), PNG, JPG</p>
          </div>

          {files.length > 0 && (
            <ul className={styles.fileList}>
              {files.map(f => (
                <li key={f.name} className={styles.fileItem}>
                  <span className={styles.fileIcon}>🗂</span>
                  <span className={styles.fileName}>{f.name}</span>
                  <span className={styles.fileSize}>{(f.size / 1024).toFixed(1)} KB</span>
                  <button className={styles.removeBtn} onClick={() => removeFile(f.name)}>✕</button>
                </li>
              ))}
            </ul>
          )}
        </section>

        {/* Step 2 — Index */}
        <section className={styles.card}>
          <h2 className={styles.cardTitle}><span className={styles.step}>2</span> Выбор индекса</h2>
          <div className={styles.indexGrid}>
            {INDICES.map(i => (
              <button
                key={i}
                className={`${styles.indexBtn} ${index === i ? styles.indexBtnActive : ''}`}
                onClick={() => setIndex(i)}
              >
                {i}
              </button>
            ))}
          </div>
          {index === 'CUSTOM' && (
            <div className={styles.formulaWrap}>
              <label className={styles.formulaLabel}>Формула (доступны: B, G, R, RE, NIR, np, eps)</label>
              <input
                className={styles.formulaInput}
                placeholder="(NIR - R) / (NIR + R + eps)"
                value={formula}
                onChange={e => setFormula(e.target.value)}
              />
            </div>
          )}
          <div className={styles.indexDesc}>
            {index === 'NDVI' && '(NIR − R) / (NIR + R) — Нормализованный вегетационный индекс'}
            {index === 'NDRE' && '(NIR − RE) / (NIR + RE) — Состояние хлорофилла'}
            {index === 'GNDVI' && '(NIR − G) / (NIR + G) — Влажность растений'}
            {index === 'EVI' && '2.5 × (NIR−R) / (NIR+6R−7.5B+1) — Улучшенная вегетация'}
            {index === 'SAVI' && '(NIR−R) / (NIR+R+0.5) × 1.5 — С поправкой на почву'}
            {index === 'NDWI' && '(G − NIR) / (G + NIR) — Влажность / водные объекты'}
            {index === 'CUSTOM' && 'Произвольная формула по каналам'}
          </div>
        </section>

        {/* Step 3 — Launch */}
        <section className={styles.card}>
          <h2 className={styles.cardTitle}><span className={styles.step}>3</span> Запуск анализа</h2>
          {error && <div className={styles.errorBox}>{error}</div>}

          <div className={styles.launchRow}>
            <button
              className={styles.launchBtn}
              onClick={handleLaunch}
              disabled={loading}
            >
              {loading ? (
                <><span className={styles.spinner} /> Выполняется...</>
              ) : (
                <><span>▶</span> Запустить анализ</>
              )}
            </button>
            {(job || files.length > 0) && (
              <button className={styles.resetBtn} onClick={handleReset}>Сбросить</button>
            )}
          </div>

          {job && (
            <div className={styles.statusBox}>
              <span className={styles.statusDot} style={{ background: statusColor }} />
              <span className={styles.statusText}>{job.progress}</span>
            </div>
          )}
        </section>

        {/* Results */}
        {job?.status === 'done' && job.results.length > 0 && (
          <section className={styles.card}>
            <h2 className={styles.cardTitle}>Результаты</h2>
            <div className={styles.resultsGrid}>
              {job.results.map((r, i) => (
                <div key={i} className={styles.resultCard}>
                  <img
                    src={`data:image/png;base64,${r.preview}`}
                    alt={r.filename}
                    className={styles.resultImg}
                  />
                  <div className={styles.resultInfo}>
                    <p className={styles.resultFilename}>{r.filename}</p>
                    <p className={styles.resultIndex}>{r.index}</p>
                    <table className={styles.statsTable}>
                      <tbody>
                        {Object.entries(r.stats).map(([k, v]) => (
                          <tr key={k}>
                            <td className={styles.statKey}>{k}</td>
                            <td className={styles.statVal}>{v}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {job?.status === 'error' && (
          <section className={styles.card}>
            <div className={styles.errorBox}>{job.progress}</div>
          </section>
        )}

        {/* LSTM Prediction */}
        <section className={styles.card}>
          <h2 className={styles.cardTitle}>
            <span className={styles.step} style={{ background: '#7c3aed' }}>AI</span>
            LSTM — Предсказание вегетационного индекса
          </h2>

          <p className={styles.lstmHint}>
            Введите временной ряд: <strong>одна строка = один шаг</strong>.
            Значения на шаге разделяйте запятой или пробелом.<br />
            Пример (1 признак): <code>0.45</code> / <code>0.47</code> / <code>0.50</code>&hellip;
            Пример (3 признака): <code>0.45, 0.12, 0.87</code>
          </p>

          <textarea
            className={styles.lstmTextarea}
            placeholder={'0.42\n0.45\n0.47\n0.50\n0.53'}
            value={lstmInput}
            onChange={e => setLstmInput(e.target.value)}
          />

          {lstmError && <div className={styles.errorBox} style={{ marginTop: 10 }}>{lstmError}</div>}

          <div className={styles.launchRow} style={{ marginTop: 12 }}>
            <button
              className={styles.launchBtn}
              style={{ background: 'linear-gradient(135deg, #7c3aed, #4c1d95)' }}
              onClick={handlePredict}
              disabled={lstmLoading}
            >
              {lstmLoading
                ? <><span className={styles.spinner} /> Вычисляется...</>
                : <><span>⚡</span> Предсказать</>}
            </button>
            {lstmResult && (
              <button className={styles.resetBtn} onClick={() => { setLstmResult(null); setLstmInput('') }}>
                Сбросить
              </button>
            )}
          </div>

          {lstmResult && (
            <div className={styles.lstmResult}>
              <div className={styles.lstmResultLabel}>Предсказанное значение</div>
              <div className={styles.lstmResultValue}>{lstmResult.prediction}</div>
              <div className={styles.lstmResultMeta}>
                window={lstmResult.seq_len} шагов · {lstmResult.input_size} признак{lstmResult.input_size === 1 ? '' : 'а'}
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
