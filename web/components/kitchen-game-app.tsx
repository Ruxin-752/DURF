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
  STEPS_PER_SECOND,
  STATIONS,
  TERRAIN_ROWS,
  chooseAiAction,
  counterKey,
  createGameState,
  itemLabel,
  startGame,
  stepGame,
  togglePause,
  type Direction,
  type GameAction,
  type GameState,
} from '@/lib/game';
import { shouldIgnoreGameHotkeys } from '@/lib/game-hotkeys';
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
  lowConfidenceRouteMessage,
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
  route: FeedbackRoute;
  phrases: VisiblePhrase[];
  outcome: string;
  trace: string;
  changedFeatures: Array<[string, number]>;
}

const LABEL_COPY = {
  Evaluative: { name: 'Evaluative', color: 'var(--berry)' },
  Imperative: { name: 'Imperative', color: 'var(--pumpkin)' },
  Descriptive: { name: 'Descriptive', color: 'var(--sage)' },
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

function gameStateSnapshot(game: GameState): Record<string, unknown> {
  return {
    status: game.status,
    tick: game.tick,
    score: game.score,
    ordersCompleted: game.ordersCompleted,
    secondsLeft: game.secondsLeft,
    player: game.player,
    partner: game.partner,
    pot: game.pot,
    counterObjects: game.counterObjects,
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
      const counterObject = game.counterObjects[counterKey({ x, y })];
      const chefs = [
        game.player.x === x && game.player.y === y ? 'player' : null,
        game.partner.x === x && game.partner.y === y ? 'partner' : null,
      ].filter(Boolean) as Array<'player' | 'partner'>;
      tiles.push(
        <div
          className={`kitchen-tile ${isCounter ? 'counter-tile' : ''} ${station ? `station-${station.kind}` : ''}`}
          key={`${x}-${y}`}
          aria-label={station?.label ?? (counterObject ? `Counter holding ${itemLabel(counterObject.item)}` : isCounter ? 'Counter' : undefined)}
        >
          {station && (
            <div className="station-mark" title={station.label}>
              <span className="station-icon" aria-hidden="true" />
              <span>{station.shortLabel}</span>
              {station.kind === 'pot' && game.pot.stage !== 'empty' && (
                <small>
                  {game.pot.stage === 'ready'
                    ? 'READY'
                    : game.pot.stage === 'filling'
                      ? `${game.pot.tomatoes}T${game.pot.onions}O`
                      : `${Math.ceil(game.pot.secondsRemaining / STEPS_PER_SECOND)}s`}
                </small>
              )}
            </div>
          )}
          {counterObject && (
            <span
              className={`counter-object item-${counterObject.item}`}
              aria-label={`${itemLabel(counterObject.item)} on counter`}
              title={`${itemLabel(counterObject.item)} on counter`}
            />
          )}
          {chefs.map((chef) => {
            const data = chef === 'player' ? game.player : game.partner;
            return (
              <div
                aria-label={`${chef === 'player' ? 'You' : 'AI partner'}, facing ${data.facing}, holding ${itemLabel(data.held)}`}
                className={`chef chef-${chef}`}
                data-facing={data.facing}
                key={chef}
                title={chef === 'player' ? 'You' : 'AI partner'}
              >
                <span className="chef-hat" />
                <span className="chef-face" />
                <span className="chef-body" />
                <span className="chef-name">{chef === 'player' ? 'YOU' : 'AI'}</span>
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
        aria-label="Cooperative kitchen game board"
        style={{
          gridTemplateColumns: `repeat(${BOARD_WIDTH}, minmax(0, 1fr))`,
          aspectRatio: `${BOARD_WIDTH} / ${BOARD_HEIGHT}`,
        }}
      >
        {tiles}
      </div>
      {game.status === 'paused' && (
        <div className="pause-overlay" role="status" aria-live="assertive">
          <strong>PAUSED</strong>
          <span>Press P or Resume to continue</span>
        </div>
      )}
      <div className="board-legend" aria-hidden="true">
        <span>Tomato T</span><i /> <span>Onion O</span><i /> <span>Pot P</span><i /> <span>Dish D</span><i /> <span>Serve S</span>
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
    <div className="controls" role="group" aria-label="Game controls">
      <div className="dpad" role="group" aria-label="Movement direction">
        <button disabled={disabled} onClick={() => onMove('up')} aria-label="Move up">▲</button>
        <button disabled={disabled} onClick={() => onMove('left')} aria-label="Move left">◀</button>
        <button disabled={disabled} onClick={() => onMove('down')} aria-label="Move down">▼</button>
        <button disabled={disabled} onClick={() => onMove('right')} aria-label="Move right">▶</button>
      </div>
      <button className="action-button" disabled={disabled} onClick={onInteract}>
        <span>SPACE</span>
        INTERACT
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
            <span>{LABEL_COPY[label].name}</span>
            <div
              aria-label={`${LABEL_COPY[label].name} probability ${(score * 100).toFixed(1)}%`}
              aria-valuemax={100}
              aria-valuemin={0}
              aria-valuenow={Number((score * 100).toFixed(1))}
              className="probability-track"
              role="progressbar"
            >
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
  const pendingHumanActionRef = useRef<GameAction>('stay');

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
        setModelError(error instanceof Error ? error.message : 'Unknown loading error');
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

  const queueHumanAction = useCallback(
    (action: GameAction, eventType: 'move' | 'interact') => {
      const current = gameRef.current;
      if (current.status !== 'running') return;
      pendingHumanActionRef.current = action;
      queueRef.current?.enqueue(eventType, {
        requestedAction: action,
        scheduledForTick: current.tick + 1,
      });
    },
    [],
  );
  const handleMove = useCallback(
    (direction: Direction) => queueHumanAction(direction, 'move'),
    [queueHumanAction],
  );
  const handleInteract = useCallback(
    () => queueHumanAction('interact', 'interact'),
    [queueHumanAction],
  );
  const handlePause = useCallback(() => {
    pendingHumanActionRef.current = 'stay';
    const type = gameRef.current.status === 'running' ? 'pause' : 'resume';
    commitGame(togglePause, type);
  }, [commitGame]);

  useEffect(() => {
    const keyDown = (event: KeyboardEvent) => {
      if (shouldIgnoreGameHotkeys(event.target)) return;
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
        if (event.repeat) return;
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
      if (previous.status !== 'running') return;
      const humanAction = pendingHumanActionRef.current;
      pendingHumanActionRef.current = 'stay';
      const aiAction = chooseAiAction(previous);
      const next = stepGame(previous, aiAction, humanAction);
      gameRef.current = next;
      setGame(next);
      queueRef.current?.enqueue('tick_summary', {
        tick: next.tick,
        requestedJointActions: { ai: aiAction, human: humanAction },
        executedJointActions: { ai: aiAction, human: humanAction },
        stateBefore: gameStateSnapshot(previous),
        stateAfter: gameStateSnapshot(next),
        reward: next.score - previous.score,
        done: next.status === 'finished',
        stepEvents: next.lastStepEvents,
        featureCounts: gameFeatureCounts(next),
      });
      if (previous.status === 'running' && next.status === 'finished') {
        queueRef.current?.end(gameSummary(next));
      }
    }, 1_000 / STEPS_PER_SECOND);
    return () => window.clearInterval(timer);
  }, [game.status]);

  const beginRound = useCallback(() => {
    if (!consented || !models || modelStatus !== 'ready') return;
    queueRef.current?.end(gameSummary(gameRef.current));
    const queue = new ResearchEventQueue(getOrCreateAnonymousUserId(), setSyncStatus);
    queueRef.current = queue;
    route1Ref.current = createFullGaussianPrior(models.features);
    route2Ref.current = createIndependentGaussianPrior(models.features);
    pendingHumanActionRef.current = 'stay';
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
          ? `Route 1 did not update the weights: ${reason ?? 'grounding_rejected'}`
          : `Route 1 updated ${results.length} phrase${results.length === 1 ? '' : 's'} through the paper-aligned pipeline`;
        trace = rejected
          ? `u → fG (low confidence/rejected: ${reason ?? 'unknown'}) → no update to w`
          : 'For each phrase: u → fG (3 classes) → f (trajectory/action/named features) → phrase-specific ζ → one Bayesian update to w. Any rejection rolls back the whole utterance.';
      } else {
        const prior = route2Ref.current ?? createIndependentGaussianPrior(models.features);
        const prediction = models.route2(utterance, gameFeatureCounts(gameRef.current));
        const result = applyRoute2Gaussian(prior, prediction.weights, 2);
        route2Ref.current = result.state;
        updaterModelHash = prediction.modelHash;
        changedFeatures = mostChanged(result.delta);
        outcome = `Route 2 used ${prediction.ensembleSize} models to update all ${models.features.length} reward weights`;
        trace = 'u + trajectory → 10-model ensemble → 53D reward vector → independent-Gaussian update with precision 2 (fG is diagnostic only).';
      }

      setVisibleFeedback({ utterance, route, phrases: visiblePhrases, outcome, trace, changedFeatures });
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
        route,
        phrases: [],
        outcome: `Feedback processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        trace: 'No model weights were changed.',
        changedFeatures: [],
      });
    } finally {
      setProcessingFeedback(false);
    }
  };

  const statusCopy = {
    waiting: 'Waiting to start',
    running: 'Shift running',
    paused: 'Paused',
    finished: 'Shift complete',
  }[game.status];

  return (
    <main className="site-shell">
      <header className="site-header">
        <div className="brand-block">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <div>
            <p>DURF RESEARCH BETA</p>
            <h1>DURF Kitchen Lab</h1>
          </div>
        </div>
        <div className="header-badges">
          <span className={`status-pill status-${game.status}`} role="status" aria-live="polite"><i />{statusCopy}</span>
          <span className={`model-pill model-${modelStatus}`} role="status" aria-live="polite">
            {modelStatus === 'loading' && 'Loading models…'}
            {modelStatus === 'ready' && 'Browser models ready'}
            {modelStatus === 'error' && 'Model loading failed'}
          </span>
        </div>
      </header>

      <section className="workspace">
        <aside className="panel mission-panel">
          <div className="panel-heading">
            <p>SHIFT BOARD</p>
            <h2>Round Mission</h2>
          </div>

          <div className="score-grid">
            <div><span>Score</span><strong>{game.score}</strong></div>
            <div><span>Time left</span><strong className={game.secondsLeft <= 15 ? 'urgent' : ''}>{game.secondsLeft}s</strong></div>
          </div>

          <div className="order-ticket">
            <div className="ticket-pin" />
            <span>Current order #{game.ordersCompleted + 1}</span>
            <div className="soup-pixel" aria-hidden="true"><i /><i /><i /></div>
            <h3>Tomato-Onion Soup</h3>
            <ol>
              <li><b>1</b> Add two tomatoes</li>
              <li><b>2</b> Add one onion</li>
              <li><b>3</b> Wait until cooked</li>
              <li><b>4</b> Plate and serve the soup</li>
            </ol>
            <p>Correct delivery reward: +20 points</p>
          </div>

          <div className="crew-status">
            <h3>Team Status</h3>
            <div><span className="mini-chef mini-player" />You <strong>{itemLabel(game.player.held)}</strong></div>
            <div><span className="mini-chef mini-ai" />AI partner <strong>{itemLabel(game.partner.held)}</strong></div>
            <div><span className={`mini-pot pot-${game.pot.stage}`} />Pot <strong>{game.pot.stage === 'empty' ? 'Empty' : game.pot.stage === 'filling' ? `${game.pot.tomatoes} tomato, ${game.pot.onions} onion` : game.pot.stage === 'ready' ? 'Ready' : `Cooking · ${Math.ceil(game.pot.secondsRemaining / STEPS_PER_SECOND)}s`}</strong></div>
          </div>

          <div className="consent-card">
            <h3>Anonymous Research Consent</h3>
            <p>After you start, we collect an anonymous UUID, game trajectories, text feedback, three-class probabilities, and model-update traces to improve the research models.</p>
            <p>
              The app does not write raw IP addresses, names, or email addresses to the research tables. The hosting platform may retain necessary operational logs.
              Do not include personal information in feedback. Close the page at any time to stop further collection.
            </p>
            <label>
              <input
                type="checkbox"
                checked={consented}
                onChange={(event) => setConsented(event.target.checked)}
                disabled={game.status !== 'waiting'}
              />
              <span>I have read this notice and voluntarily consent to anonymous research data collection.</span>
            </label>
            <button
              className="primary-button"
              onClick={beginRound}
              disabled={!consented || modelStatus !== 'ready' || game.status !== 'waiting'}
            >
              Start Shift
            </button>
            {modelStatus === 'error' && (
              <button className="text-button" onClick={() => window.location.reload()}>
                Retry model loading
              </button>
            )}
          </div>

          <div className={`sync-line sync-${syncStatus.state}`} role="status" aria-live="polite">
            <i /> Data queue: {syncStatus.state === 'synced' ? 'Synced' : syncStatus.state === 'syncing' ? 'Syncing' : syncStatus.state === 'offline' ? 'Offline — retrying' : `${syncStatus.pending} pending`}
          </div>
        </aside>

        <section className="game-column">
          <div className="game-toolbar">
            <div role="status" aria-live="polite" aria-atomic="true">
              <span className="live-dot" />
              {game.lastAction}
            </div>
            <div className="toolbar-actions">
              <button onClick={handlePause} disabled={!['running', 'paused'].includes(game.status)}>
                {game.status === 'paused' ? 'Resume' : 'Pause'}
              </button>
              <button onClick={beginRound} disabled={!consented || modelStatus !== 'ready'}>
                Restart
              </button>
            </div>
          </div>

          {modelStatus === 'loading' && (
            <div className="model-notice" role="status" aria-live="polite"><span className="pixel-loader" />Loading the three-class model and Route 2 ensemble…</div>
          )}
          {modelStatus === 'error' && (
            <div className="model-notice notice-error" role="alert">Model files could not be loaded: {modelError}</div>
          )}

          <KitchenBoard game={game} />
          {game.status === 'finished' && (
            <div className="round-result" role="status">
              <span>SHIFT COMPLETE</span>
              <strong>{game.score} points</strong>
              <p>{game.ordersCompleted} order{game.ordersCompleted === 1 ? '' : 's'} completed</p>
              <button onClick={beginRound}>Play again</button>
            </div>
          )}

          <Controls
            disabled={game.status !== 'running'}
            onMove={handleMove}
            onInteract={handleInteract}
          />
          <p className="keyboard-help"><kbd>WASD</kbd> / <kbd>Arrow keys</kbd> move · <kbd>Space</kbd> interact · <kbd>P</kbd> pause</p>
        </section>

        <aside className="panel feedback-panel">
          <div className="panel-heading">
            <p>LANGUAGE FEEDBACK</p>
            <h2>Say Something to the AI</h2>
          </div>

          <div className="route-switch" role="radiogroup" aria-label="Feedback update route">
            <button className={route === 'route1' ? 'active' : ''} onClick={() => setRoute('route1')} role="radio" aria-checked={route === 'route1'}>
              <b>Route 1</b><span>Paper-aligned decoupled pipeline</span>
            </button>
            <button className={route === 'route2' ? 'active' : ''} onClick={() => setRoute('route2')} role="radio" aria-checked={route === 'route2'}>
              <b>Route 2</b><span>10-model ensemble</span>
            </button>
          </div>

          <form className="feedback-form" onSubmit={submitFeedback}>
            <label htmlFor="feedback-input">Feedback (up to 500 characters)</label>
            <textarea
              id="feedback-input"
              maxLength={500}
              value={feedbackText}
              onChange={(event) => setFeedbackText(event.target.value)}
              placeholder="Example: That last route was bad. Please take a dish instead."
              disabled={game.status === 'waiting' || modelStatus !== 'ready'}
            />
            <div><span>{feedbackText.length}/500</span><button disabled={!feedbackText.trim() || processingFeedback || game.status === 'waiting'}>{processingFeedback ? 'Analyzing…' : 'Analyze and update'}</button></div>
          </form>

          <div className="classifier-key">
            {(Object.keys(LABEL_COPY) as Array<keyof typeof LABEL_COPY>).map((label) => (
              <span key={label}><i style={{ background: LABEL_COPY[label].color }} />{LABEL_COPY[label].name}</span>
            ))}
          </div>

          {!visibleFeedback && (
            <div className="empty-feedback">
              <div className="speech-pixels" aria-hidden="true"><i /><i /><i /></div>
              <h3>Phrase-by-phrase classifications appear here</h3>
              <p>Each phrase shows all three probabilities, the top class, calibrated confidence, and any low-confidence warning.</p>
            </div>
          )}

          {visibleFeedback && (
            <div className="feedback-results" role="status" aria-live="polite">
              {visibleFeedback.phrases.map((phrase, index) => {
                const copy = LABEL_COPY[phrase.research.label];
                return (
                  <article className={`phrase-card ${phrase.model.abstained ? 'low-confidence' : ''}`} key={`${phrase.research.phrase}-${index}`}>
                    <div className="phrase-number">Phrase {index + 1}</div>
                    <blockquote>{phrase.research.phrase}</blockquote>
                    <div className="prediction-line">
                      <span style={{ color: copy.color }}>{copy.name}</span>
                      <strong>{(phrase.model.confidence * 100).toFixed(1)}%</strong>
                    </div>
                    <ProbabilityRows phrase={phrase} />
                    <p className={phrase.model.abstained ? 'confidence-warning' : 'confidence-ok'}>
                      {phrase.model.abstained
                        ? lowConfidenceRouteMessage(visibleFeedback.route, phrase.model.threshold)
                        : `Calibrated confidence · threshold ${(phrase.model.threshold * 100).toFixed(0)}%`}
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
            <summary>Data and privacy</summary>
            <p>
              Events are batched to D1. Idempotent event IDs prevent duplicate writes, and offline events stay in this page&apos;s memory queue for retry.
              Records include schema/model hashes, full probabilities, and route traces. Abuse protection uses only a short-lived network-address hash.
              Raw IP addresses are not stored or exported, and the research export endpoint requires an administrator Bearer token.
            </p>
          </details>
        </aside>
      </section>

      <footer>
        <span>DURF Kitchen Lab · Research Beta</span>
        <span>Original CSS pixel art · No Overcooked or Team17 assets used</span>
      </footer>
    </main>
  );
}
