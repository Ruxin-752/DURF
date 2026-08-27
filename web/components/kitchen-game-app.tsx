'use client';

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from 'react';
import {
  applyRoute1PaperUpdate,
  applyRoute2Gaussian,
  createFullGaussianPrior,
  createIndependentGaussianPrior,
  loadBrowserModels,
  type BrowserModels,
  type FeedbackFormPrediction,
  type FullGaussianState,
  type IndependentGaussianState,
  type Route1PaperResult,
} from '@/lib/browser-models';
import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  STATIONS,
  TERRAIN_ROWS,
  createGameState,
  interactPlayer,
  itemLabel,
  movePlayer,
  startGame,
  tickGame,
  togglePause,
  type Direction,
  type GameState,
} from '@/lib/game';
import {
  ResearchEventQueue,
  getOrCreateAnonymousUserId,
  type QueueStatus,
} from '@/lib/research-client';
import {
  SCHEMA_VERSION,
  type FeedbackRoute,
  type PhraseResearchPrediction,
} from '@/lib/research-types';
import {
  actionFeaturesFromText,
  aggregateProbabilities,
  gameFeatureCounts,
  inferValence,
  namedFeaturesFromText,
  splitFeedbackPhrases,
  toResearchPrediction,
  topResearchLabel,
} from '@/lib/route-inputs';

interface VisiblePhrase {
  model: FeedbackFormPrediction;
  research: PhraseResearchPrediction;
}

interface VisibleFeedback {
  utterance: string;
  phrases: VisiblePhrase[];
  outcome: string;
  trace: string;
  changedFeatures: Array<[string, number]>;
}

const LABEL_COPY = {
  Evaluative: { name: 'Evaluative', chinese: '评价型', color: 'var(--berry)' },
  Imperative: { name: 'Imperative', chinese: '指令型', color: 'var(--pumpkin)' },
  Descriptive: { name: 'Descriptive', chinese: '描述型', color: 'var(--sage)' },
} as const;

const EMPTY_SYNC: QueueStatus = { pending: 0, state: 'idle' };

function gameSummary(game: GameState): Record<string, unknown> {
  return {
    score: game.score,
    ordersCompleted: game.ordersCompleted,
    secondsRemaining: game.secondsLeft,
    finalStatus: game.status,
    finalTick: game.tick,
  };
}

function mostChanged(delta: Record<string, number>): Array<[string, number]> {
  return Object.entries(delta)
    .filter(([, value]) => Number.isFinite(value))
    .sort((left, right) => Math.abs(right[1]) - Math.abs(left[1]))
    .slice(0, 5);
}

function stationAt(x: number, y: number) {
  return STATIONS.find(
    (station) => station.position.x === x && station.position.y === y,
  );
}

function KitchenBoard({ game }: { game: GameState }) {
  const tiles = [];
  for (let y = 0; y < BOARD_HEIGHT; y += 1) {
    for (let x = 0; x < BOARD_WIDTH; x += 1) {
      const station = stationAt(x, y);
      const terrain = TERRAIN_ROWS[y][x];
      const isCounter = terrain === 'X';
      const chefs = [
        game.player.x === x && game.player.y === y ? 'player' : null,
        game.partner.x === x && game.partner.y === y ? 'partner' : null,
      ].filter(Boolean) as Array<'player' | 'partner'>;
      tiles.push(
        <div
          className={`kitchen-tile ${isCounter ? 'counter-tile' : ''} ${station ? `station-${station.kind}` : ''}`}
          key={`${x}-${y}`}
          aria-label={station?.label}
        >
          {station && (
            <div className="station-mark" title={station.label}>
              <span className="station-icon" aria-hidden="true" />
              <span>{station.shortLabel}</span>
              {station.kind === 'pot' && game.pot.stage !== 'empty' && (
                <small>
                  {game.pot.stage === 'ready'
                    ? '好了'
                    : game.pot.stage === 'filling'
                      ? `${game.pot.tomatoes}T${game.pot.onions}O`
                      : `${game.pot.secondsRemaining}s`}
                </small>
              )}
            </div>
          )}
          {chefs.map((chef) => {
            const data = chef === 'player' ? game.player : game.partner;
            return (
              <div className={`chef chef-${chef}`} key={chef} title={chef === 'player' ? '你' : 'AI伙伴'}>
                <span className="chef-hat" />
                <span className="chef-face" />
                <span className="chef-body" />
                <span className="chef-name">{chef === 'player' ? '你' : 'AI'}</span>
                {data.held && <span className={`held-item item-${data.held}`} />}
              </div>
            );
          })}
        </div>,
      );
    }
  }
  return (
    <div className="board-frame">
      <div className="awning" aria-hidden="true" />
      <div
        className="kitchen-board"
        role="img"
        aria-label="合作厨房游戏地图"
        style={{
          gridTemplateColumns: `repeat(${BOARD_WIDTH}, minmax(0, 1fr))`,
          aspectRatio: `${BOARD_WIDTH} / ${BOARD_HEIGHT}`,
        }}
      >
        {tiles}
      </div>
      <div className="board-legend" aria-hidden="true">
        <span>番茄 T</span><i /> <span>洋葱 O</span><i /> <span>锅 P</span><i /> <span>盘 D</span><i /> <span>出餐 S</span>
      </div>
    </div>
  );
}

function Controls({
  disabled,
  onMove,
  onInteract,
}: {
  disabled: boolean;
  onMove: (direction: Direction) => void;
  onInteract: () => void;
}) {
  return (
    <div className="controls" aria-label="游戏控制">
      <div className="dpad">
        <button disabled={disabled} onClick={() => onMove('up')} aria-label="向上">▲</button>
        <button disabled={disabled} onClick={() => onMove('left')} aria-label="向左">◀</button>
        <button disabled={disabled} onClick={() => onMove('down')} aria-label="向下">▼</button>
        <button disabled={disabled} onClick={() => onMove('right')} aria-label="向右">▶</button>
      </div>
      <button className="action-button" disabled={disabled} onClick={onInteract}>
        <span>SPACE</span>
        操作
      </button>
    </div>
  );
}

function ProbabilityRows({ phrase }: { phrase: VisiblePhrase }) {
  return (
    <div className="probability-list">
      {(Object.keys(LABEL_COPY) as Array<keyof typeof LABEL_COPY>).map((label) => {
        const score = phrase.research.probabilities[label];
        return (
          <div className="probability-row" key={label}>
            <span>{label.slice(0, 4)}</span>
            <div className="probability-track">
              <i style={{ width: `${Math.max(1, score * 100)}%`, background: LABEL_COPY[label].color }} />
            </div>
            <strong>{(score * 100).toFixed(1)}%</strong>
          </div>
        );
      })}
    </div>
  );
}

export function KitchenGameApp() {
  const [game, setGame] = useState<GameState>(() => createGameState());
  const gameRef = useRef(game);
  const [consented, setConsented] = useState(false);
  const [models, setModels] = useState<BrowserModels | null>(null);
  const [modelStatus, setModelStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [modelError, setModelError] = useState('');
  const [route, setRoute] = useState<FeedbackRoute>('route1');
  const [feedbackText, setFeedbackText] = useState('');
  const [visibleFeedback, setVisibleFeedback] = useState<VisibleFeedback | null>(null);
  const [processingFeedback, setProcessingFeedback] = useState(false);
  const [syncStatus, setSyncStatus] = useState<QueueStatus>(EMPTY_SYNC);
  const queueRef = useRef<ResearchEventQueue | null>(null);
  const route1Ref = useRef<FullGaussianState | null>(null);
  const route2Ref = useRef<IndependentGaussianState | null>(null);

  useEffect(() => {
    let cancelled = false;
    loadBrowserModels()
      .then((loaded) => {
        if (cancelled) return;
        setModels(loaded);
        route1Ref.current = createFullGaussianPrior(loaded.features);
        route2Ref.current = createIndependentGaussianPrior(loaded.features);
        setModelStatus('ready');
      })
      .catch((error) => {
        if (cancelled) return;
        setModelError(error instanceof Error ? error.message : '未知加载错误');
        setModelStatus('error');
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const handlePageHide = () => queueRef.current?.flushWithBeacon();
    window.addEventListener('pagehide', handlePageHide);
    return () => window.removeEventListener('pagehide', handlePageHide);
  }, []);

  const commitGame = useCallback(
    (
      update: (current: GameState) => GameState,
      eventType?: 'move' | 'interact' | 'pause' | 'resume',
    ) => {
      const previous = gameRef.current;
      const next = update(previous);
      if (next === previous) return;
      gameRef.current = next;
      setGame(next);
      if (eventType) {
        queueRef.current?.enqueue(eventType, {
          action: next.lastAction,
          player: { x: next.player.x, y: next.player.y, held: next.player.held },
          partner: { x: next.partner.x, y: next.partner.y, held: next.partner.held },
          pot: next.pot,
          score: next.score,
          tick: next.tick,
        });
      }
    },
    [],
  );

  const handleMove = useCallback(
    (direction: Direction) => commitGame((current) => movePlayer(current, direction), 'move'),
    [commitGame],
  );
  const handleInteract = useCallback(
    () => commitGame(interactPlayer, 'interact'),
    [commitGame],
  );
  const handlePause = useCallback(() => {
    const type = gameRef.current.status === 'running' ? 'pause' : 'resume';
    commitGame(togglePause, type);
  }, [commitGame]);

  useEffect(() => {
    const keyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.matches('input, textarea, select, button')) return;
      const directions: Record<string, Direction> = {
        w: 'up',
        arrowup: 'up',
        a: 'left',
        arrowleft: 'left',
        s: 'down',
        arrowdown: 'down',
        d: 'right',
        arrowright: 'right',
      };
      const key = event.key.toLowerCase();
      if (directions[key]) {
        event.preventDefault();
        handleMove(directions[key]);
      } else if (event.code === 'Space') {
        event.preventDefault();
        handleInteract();
      } else if (key === 'p') {
        event.preventDefault();
        handlePause();
      }
    };
    window.addEventListener('keydown', keyDown);
    return () => window.removeEventListener('keydown', keyDown);
  }, [handleInteract, handleMove, handlePause]);

  useEffect(() => {
    if (game.status !== 'running') return;
    const timer = window.setInterval(() => {
      const previous = gameRef.current;
      const next = tickGame(previous);
      gameRef.current = next;
      setGame(next);
      if (next.tick % 5 === 0 || next.status === 'finished') {
        queueRef.current?.enqueue('tick_summary', {
          tick: next.tick,
          secondsLeft: next.secondsLeft,
          score: next.score,
          ordersCompleted: next.ordersCompleted,
          player: { x: next.player.x, y: next.player.y, held: next.player.held },
          partner: { x: next.partner.x, y: next.partner.y, held: next.partner.held },
          pot: next.pot,
          featureCounts: gameFeatureCounts(next),
        });
      }
      if (previous.status === 'running' && next.status === 'finished') {
        queueRef.current?.end(gameSummary(next));
      }
    }, 1_000);
    return () => window.clearInterval(timer);
  }, [game.status]);

  const beginRound = useCallback(() => {
    if (!consented || !models || modelStatus !== 'ready') return;
    queueRef.current?.end(gameSummary(gameRef.current));
    const queue = new ResearchEventQueue(getOrCreateAnonymousUserId(), setSyncStatus);
    queueRef.current = queue;
    route1Ref.current = createFullGaussianPrior(models.features);
    route2Ref.current = createIndependentGaussianPrior(models.features);
    const next = startGame(createGameState());
    gameRef.current = next;
    setGame(next);
    setVisibleFeedback(null);
    setFeedbackText('');
  }, [consented, modelStatus, models]);

  const submitFeedback = async (event: FormEvent) => {
    event.preventDefault();
    const utterance = feedbackText.trim();
    if (!utterance || !models || game.status === 'waiting') return;
    const phrases = splitFeedbackPhrases(utterance);
    if (phrases.length === 0) return;
    setProcessingFeedback(true);
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

    try {
      const modelPredictions = phrases.map((phrase) => models.classify(phrase));
      const visiblePhrases = modelPredictions.map((prediction, index) => ({
        model: prediction,
        research: toResearchPrediction(phrases[index], prediction),
      }));
      const aggregate = aggregateProbabilities(modelPredictions);
      const lowConfidence = modelPredictions.some((prediction) => prediction.abstained);
      const classifierHash = modelPredictions[0].modelHash;
      let trace = '';
      let outcome = '';
      let changedFeatures: Array<[string, number]> = [];
      let updaterModelHash = classifierHash;

      if (route === 'route1') {
        const prior = route1Ref.current ?? createFullGaussianPrior(models.features);
        let candidate = prior;
        const results: Route1PaperResult[] = [];
        if (!lowConfidence) {
          for (const [index, prediction] of modelPredictions.entries()) {
            const phraseText = phrases[index];
            const result = applyRoute1PaperUpdate(candidate, {
              feedbackForm: prediction.label,
              feedbackFormConfidence: prediction.confidence,
              feedbackFormThreshold: prediction.threshold,
              feedbackFormAbstained: prediction.abstained,
              trajectoryFeatures: gameFeatureCounts(gameRef.current),
              actionFeatures: actionFeaturesFromText(phraseText),
              namedFeatures: namedFeaturesFromText(phraseText),
              valence: inferValence(phraseText),
              basePrecision: 2,
            });
            results.push(result);
            candidate = result.state;
          }
        }
        const rejected =
          lowConfidence || results.some((result) => result.status === 'rejected');
        if (!rejected) route1Ref.current = candidate;
        const mergedDelta = rejected
          ? {}
          : Object.fromEntries(
              models.features.map((feature, index) => [
                feature,
                candidate.mean[index] - prior.mean[index],
              ]),
            );
        changedFeatures = mostChanged(mergedDelta);
        const reason = lowConfidence
          ? 'feedback_form_low_confidence'
          : results.find((result) => result.status === 'rejected')?.reason;
        outcome = rejected
          ? `Route1 未改权重：${reason ?? 'grounding_rejected'}`
          : `Route1 已按论文链路更新 ${results.length} 个短语`;
        trace = rejected
          ? `u → fG(低置信/拒绝:${reason ?? 'unknown'}) → 不更新 w`
          : '每个短语 u → fG(三类) → f(轨迹/动作/命名特征) → 各自 ζ → 统一贝叶斯更新 w；任一拒绝则整句回滚';
      } else {
        const prior = route2Ref.current ?? createIndependentGaussianPrior(models.features);
        const prediction = models.route2(utterance, gameFeatureCounts(gameRef.current));
        const result = applyRoute2Gaussian(prior, prediction.weights, 2);
        route2Ref.current = result.state;
        updaterModelHash = prediction.modelHash;
        changedFeatures = mostChanged(result.delta);
        outcome = `Route2 的 ${prediction.ensembleSize} 个模型已更新全部 ${models.features.length} 维权重`;
        trace = 'u + trajectory → 10模型集成 → 53维奖励向量 → 独立高斯精度2更新 w（fG仅诊断）';
      }

      setVisibleFeedback({ utterance, phrases: visiblePhrases, outcome, trace, changedFeatures });
      const feedbackId = crypto.randomUUID();
      queueRef.current?.enqueue(
        'feedback',
        {
          utterance,
          selectedRoute: route,
          classifierModelHash: classifierHash,
          updaterModelHash,
          calibrated: modelPredictions.every((prediction) => prediction.calibrated),
          calibrationVersions: modelPredictions.map(
            (prediction) => prediction.calibrationVersion,
          ),
          outcome,
          changedFeatures: Object.fromEntries(changedFeatures),
          gameSnapshot: {
            tick: gameRef.current.tick,
            score: gameRef.current.score,
            features: gameFeatureCounts(gameRef.current),
          },
        },
        {
          modelHash: updaterModelHash,
          routeTrace: trace,
          probabilities: aggregate,
          feedback: {
            feedbackId,
            utterance,
            route,
            topLabel: topResearchLabel(aggregate),
            lowConfidence,
            probabilities: aggregate,
            phrases: visiblePhrases.map((phrase) => phrase.research),
            modelHash: classifierHash,
            schemaVersion: SCHEMA_VERSION,
            routeTrace: trace,
          },
        },
      );
      setFeedbackText('');
    } catch (error) {
      setVisibleFeedback({
        utterance,
        phrases: [],
        outcome: `反馈处理失败：${error instanceof Error ? error.message : '未知错误'}`,
        trace: '未写入模型权重',
        changedFeatures: [],
      });
    } finally {
      setProcessingFeedback(false);
    }
  };

  const statusCopy = {
    waiting: '等待开始',
    running: '营业中',
    paused: '已暂停',
    finished: '本轮结束',
  }[game.status];

  return (
    <main className="site-shell">
      <header className="site-header">
        <div className="brand-block">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <div>
            <p>DURF RESEARCH BETA</p>
            <h1>暖炉厨房实验室</h1>
          </div>
        </div>
        <div className="header-badges">
          <span className={`status-pill status-${game.status}`}><i />{statusCopy}</span>
          <span className={`model-pill model-${modelStatus}`}>
            {modelStatus === 'loading' && '模型加载中…'}
            {modelStatus === 'ready' && '浏览器模型就绪'}
            {modelStatus === 'error' && '模型加载失败'}
          </span>
        </div>
      </header>

      <section className="workspace">
        <aside className="panel mission-panel">
          <div className="panel-heading">
            <p>SHIFT BOARD</p>
            <h2>本轮任务</h2>
          </div>

          <div className="score-grid">
            <div><span>得分</span><strong>{game.score}</strong></div>
            <div><span>剩余</span><strong className={game.secondsLeft <= 15 ? 'urgent' : ''}>{game.secondsLeft}s</strong></div>
          </div>

          <div className="order-ticket">
            <div className="ticket-pin" />
            <span>当前订单 #{game.ordersCompleted + 1}</span>
            <div className="soup-pixel" aria-hidden="true"><i /><i /><i /></div>
            <h3>番茄洋葱汤</h3>
            <ol>
              <li><b>1</b> 两个番茄入锅</li>
              <li><b>2</b> 一个洋葱入锅</li>
              <li><b>3</b> 等待烹煮完成</li>
              <li><b>4</b> 拿盘盛汤并出餐</li>
            </ol>
            <p>完成奖励 +100 分 · 加时 5 秒</p>
          </div>

          <div className="crew-status">
            <h3>协作状态</h3>
            <div><span className="mini-chef mini-player" />你 <strong>{itemLabel(game.player.held)}</strong></div>
            <div><span className="mini-chef mini-ai" />AI伙伴 <strong>{itemLabel(game.partner.held)}</strong></div>
            <div><span className={`mini-pot pot-${game.pot.stage}`} />汤锅 <strong>{game.pot.stage === 'empty' ? '空' : game.pot.stage === 'filling' ? `${game.pot.tomatoes}番茄 ${game.pot.onions}洋葱` : game.pot.stage === 'ready' ? '已煮好' : `烹煮 ${game.pot.secondsRemaining}s`}</strong></div>
          </div>

          <div className="consent-card">
            <h3>匿名研究同意</h3>
            <p>开始后仅收集匿名 UUID、游戏轨迹、文字反馈、三类概率与模型更新路径，用于改进研究模型。</p>
            <p><strong>不收集</strong> IP、姓名或邮箱。可随时关闭页面退出。</p>
            <label>
              <input
                type="checkbox"
                checked={consented}
                onChange={(event) => setConsented(event.target.checked)}
                disabled={game.status !== 'waiting'}
              />
              <span>我已阅读并自愿同意匿名研究数据收集</span>
            </label>
            <button
              className="primary-button"
              onClick={beginRound}
              disabled={!consented || modelStatus !== 'ready' || game.status !== 'waiting'}
            >
              开始营业
            </button>
            {modelStatus === 'error' && (
              <button className="text-button" onClick={() => window.location.reload()}>
                重试加载模型
              </button>
            )}
          </div>

          <div className={`sync-line sync-${syncStatus.state}`}>
            <i /> 数据队列：{syncStatus.state === 'synced' ? '已同步' : syncStatus.state === 'syncing' ? '同步中' : syncStatus.state === 'offline' ? '离线等待重试' : `${syncStatus.pending} 条待发送`}
          </div>
        </aside>

        <section className="game-column">
          <div className="game-toolbar">
            <div>
              <span className="live-dot" />
              {game.lastAction}
            </div>
            <div className="toolbar-actions">
              <button onClick={handlePause} disabled={!['running', 'paused'].includes(game.status)}>
                {game.status === 'paused' ? '继续' : '暂停'}
              </button>
              <button onClick={beginRound} disabled={!consented || modelStatus !== 'ready'}>
                重新开局
              </button>
            </div>
          </div>

          {modelStatus === 'loading' && (
            <div className="model-notice"><span className="pixel-loader" />正在加载三分类与 Route2 十模型集成，请稍候…</div>
          )}
          {modelStatus === 'error' && (
            <div className="model-notice notice-error">模型文件未能加载：{modelError}</div>
          )}

          <KitchenBoard game={game} />
          {game.status === 'finished' && (
            <div className="round-result" role="status">
              <span>SHIFT COMPLETE</span>
              <strong>{game.score} 分</strong>
              <p>完成 {game.ordersCompleted} 份订单</p>
              <button onClick={beginRound}>再来一轮</button>
            </div>
          )}

          <Controls
            disabled={game.status !== 'running'}
            onMove={handleMove}
            onInteract={handleInteract}
          />
          <p className="keyboard-help"><kbd>WASD</kbd> / <kbd>方向键</kbd> 移动 · <kbd>Space</kbd> 操作 · <kbd>P</kbd> 暂停</p>
        </section>

        <aside className="panel feedback-panel">
          <div className="panel-heading">
            <p>LANGUAGE FEEDBACK</p>
            <h2>对 AI 说一句话</h2>
          </div>

          <div className="route-switch" role="radiogroup" aria-label="反馈更新路线">
            <button className={route === 'route1' ? 'active' : ''} onClick={() => setRoute('route1')} role="radio" aria-checked={route === 'route1'}>
              <b>Route 1</b><span>论文解耦链路</span>
            </button>
            <button className={route === 'route2' ? 'active' : ''} onClick={() => setRoute('route2')} role="radio" aria-checked={route === 'route2'}>
              <b>Route 2</b><span>10 模型集成</span>
            </button>
          </div>

          <form className="feedback-form" onSubmit={submitFeedback}>
            <label htmlFor="feedback-input">反馈内容（最多 500 字）</label>
            <textarea
              id="feedback-input"
              maxLength={500}
              value={feedbackText}
              onChange={(event) => setFeedbackText(event.target.value)}
              placeholder="例如：That last route was bad. Please take a dish instead."
              disabled={game.status === 'waiting' || modelStatus !== 'ready'}
            />
            <div><span>{feedbackText.length}/500</span><button disabled={!feedbackText.trim() || processingFeedback || game.status === 'waiting'}>{processingFeedback ? '分析中…' : '分析并更新'}</button></div>
          </form>

          <div className="classifier-key">
            {(Object.keys(LABEL_COPY) as Array<keyof typeof LABEL_COPY>).map((label) => (
              <span key={label}><i style={{ background: LABEL_COPY[label].color }} />{LABEL_COPY[label].name}<small>{LABEL_COPY[label].chinese}</small></span>
            ))}
          </div>

          {!visibleFeedback && (
            <div className="empty-feedback">
              <div className="speech-pixels" aria-hidden="true"><i /><i /><i /></div>
              <h3>逐短语分类会显示在这里</h3>
              <p>每个短语都显示三类完整概率、最高类别、校准置信度和低置信提示。</p>
            </div>
          )}

          {visibleFeedback && (
            <div className="feedback-results" aria-live="polite">
              {visibleFeedback.phrases.map((phrase, index) => {
                const copy = LABEL_COPY[phrase.research.label];
                return (
                  <article className={`phrase-card ${phrase.model.abstained ? 'low-confidence' : ''}`} key={`${phrase.research.phrase}-${index}`}>
                    <div className="phrase-number">短语 {index + 1}</div>
                    <blockquote>{phrase.research.phrase}</blockquote>
                    <div className="prediction-line">
                      <span style={{ color: copy.color }}>{copy.name} · {copy.chinese}</span>
                      <strong>{(phrase.model.confidence * 100).toFixed(1)}%</strong>
                    </div>
                    <ProbabilityRows phrase={phrase} />
                    <p className={phrase.model.abstained ? 'confidence-warning' : 'confidence-ok'}>
                      {phrase.model.abstained
                        ? `低于 ${(phrase.model.threshold * 100).toFixed(0)}% 阈值：Route1 不更新`
                        : `已校准置信度 · 阈值 ${(phrase.model.threshold * 100).toFixed(0)}%`}
                    </p>
                  </article>
                );
              })}
              <div className="update-card">
                <span>UPDATE TRACE</span>
                <h3>{visibleFeedback.outcome}</h3>
                <p>{visibleFeedback.trace}</p>
                {visibleFeedback.changedFeatures.length > 0 && (
                  <div className="feature-chips">
                    {visibleFeedback.changedFeatures.map(([feature, value]) => (
                      <code key={feature}>{feature} {value >= 0 ? '+' : ''}{value.toFixed(3)}</code>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          <details className="privacy-details">
            <summary>数据与隐私说明</summary>
            <p>事件按批次发送到 D1；event_id 幂等，断网时留在当前页面队列重试。数据包括 schema/model hash、完整概率和 route trace。研究导出接口仅接受管理员 Bearer Token。</p>
          </details>
        </aside>
      </section>

      <footer>
        <span>DURF Kitchen Lab · Research Beta</span>
        <span>原创 CSS 像素美术 · 未使用 Overcooked/Team17 素材</span>
      </footer>
    </main>
  );
}
