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
  GAME_STEP_INTERVAL_MS,
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
import { configuredFeedbackRoute } from '@/lib/feedback-route-config';
import {
  humanActionForJointStep,
  movementDirectionForKey,
  shouldIgnoreGameHotkeys,
} from '@/lib/game-hotkeys';
import {
  ResearchEventQueue,
  getOrCreateAnonymousUserId,
} from '@/lib/research-client';
import {
  SCHEMA_VERSION,
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
  phrases: VisiblePhrase[];
  error?: string;
}

const LABEL_COPY = {
  Evaluative: { name: 'Evaluative', color: 'var(--berry)' },
  Imperative: { name: 'Imperative', color: 'var(--pumpkin)' },
  Descriptive: { name: 'Descriptive', color: 'var(--sage)' },
} as const;

const DEPLOYMENT_FEEDBACK_ROUTE = configuredFeedbackRoute();

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

function lowConfidenceCopy(threshold: number): string {
  return `Below the calibrated ${(threshold * 100).toFixed(0)}% threshold. Treat this classification as uncertain.`;
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
                <span aria-hidden="true" className="chef-feet" />
                <span aria-hidden="true" className="chef-body" />
                <span aria-hidden="true" className="chef-arms" />
                <span aria-hidden="true" className="chef-apron" />
                <span aria-hidden="true" className="chef-hair" />
                <span aria-hidden="true" className="chef-face" />
                <span aria-hidden="true" className="chef-scarf" />
                <span aria-hidden="true" className="chef-hat" />
                <span className="chef-name">{chef === 'player' ? 'YOU' : 'AI'}</span>
                {data.held && <span aria-hidden="true" className={`held-item item-${data.held}`} />}
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
  const [consentChecked, setConsentChecked] = useState(false);
  const [models, setModels] = useState<BrowserModels | null>(null);
  const [modelStatus, setModelStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [modelError, setModelError] = useState('');
  const [feedbackText, setFeedbackText] = useState('');
  const [visibleFeedback, setVisibleFeedback] = useState<VisibleFeedback | null>(null);
  const [processingFeedback, setProcessingFeedback] = useState(false);
  const queueRef = useRef<ResearchEventQueue | null>(null);
  const route1Ref = useRef<FullGaussianState | null>(null);
  const route2Ref = useRef<IndependentGaussianState | null>(null);
  const pendingHumanActionRef = useRef<GameAction>('stay');
  const pressedHumanMovementKeysRef = useRef<Set<string>>(new Set());

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
    const handlePageHide = (event: PageTransitionEvent) => {
      const queue = queueRef.current;
      if (!queue) return;
      if (!event.persisted) queue.end(gameSummary(gameRef.current));
      queue.flushWithBeacon();
    };
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
    pressedHumanMovementKeysRef.current.clear();
    const type = gameRef.current.status === 'running' ? 'pause' : 'resume';
    commitGame(togglePause, type);
  }, [commitGame]);

  useEffect(() => {
    const keyDown = (event: KeyboardEvent) => {
      if (shouldIgnoreGameHotkeys(event.target)) return;
      const key = event.key.toLowerCase();
      const direction = movementDirectionForKey(key);
      if (direction) {
        event.preventDefault();
        if (!event.repeat) {
          pressedHumanMovementKeysRef.current.delete(key);
          pressedHumanMovementKeysRef.current.add(key);
          handleMove(direction);
        }
      } else if (event.code === 'Space') {
        event.preventDefault();
        handleInteract();
      } else if (key === 'p') {
        event.preventDefault();
        if (event.repeat) return;
        handlePause();
      }
    };
    const keyUp = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      if (movementDirectionForKey(key)) {
        pressedHumanMovementKeysRef.current.delete(key);
      }
    };
    const clearHeldMovement = () => pressedHumanMovementKeysRef.current.clear();
    window.addEventListener('keydown', keyDown);
    window.addEventListener('keyup', keyUp);
    window.addEventListener('blur', clearHeldMovement);
    return () => {
      window.removeEventListener('keydown', keyDown);
      window.removeEventListener('keyup', keyUp);
      window.removeEventListener('blur', clearHeldMovement);
    };
  }, [handleInteract, handleMove, handlePause]);

  useEffect(() => {
    if (game.status !== 'running') return;
    const timer = window.setInterval(() => {
      const previous = gameRef.current;
      if (previous.status !== 'running') return;
      const humanAction = humanActionForJointStep(
        pendingHumanActionRef.current,
        pressedHumanMovementKeysRef.current,
      );
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
    }, GAME_STEP_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [game.status]);

  const startRound = useCallback((consentedAt?: number) => {
    if (!models || modelStatus !== 'ready') return;
    if (!queueRef.current) {
      if (consentedAt === undefined) return;
      queueRef.current = new ResearchEventQueue(
        getOrCreateAnonymousUserId(),
        undefined,
        undefined,
        consentedAt,
      );
    } else {
      queueRef.current.enqueue('restart', gameSummary(gameRef.current));
    }
    route1Ref.current = createFullGaussianPrior(models.features);
    route2Ref.current = createIndependentGaussianPrior(models.features);
    pendingHumanActionRef.current = 'stay';
    pressedHumanMovementKeysRef.current.clear();
    const next = startGame(createGameState());
    gameRef.current = next;
    setGame(next);
    setVisibleFeedback(null);
    setFeedbackText('');
  }, [modelStatus, models]);

  const beginRound = useCallback(() => {
    if (!consented) return;
    startRound();
  }, [consented, startRound]);

  const acceptConsentAndStart = useCallback(() => {
    if (!consentChecked) return;
    const consentedAt = Date.now();
    setConsented(true);
    startRound(consentedAt);
  }, [consentChecked, startRound]);

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

      if (DEPLOYMENT_FEEDBACK_ROUTE === 'route1') {
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
      setVisibleFeedback({ phrases: visiblePhrases });
      const feedbackId = crypto.randomUUID();
      queueRef.current?.enqueue(
        'feedback',
        {
          utterance,
          selectedRoute: DEPLOYMENT_FEEDBACK_ROUTE,
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
            route: DEPLOYMENT_FEEDBACK_ROUTE,
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
        phrases: [],
        error: `Feedback processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
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
            {modelStatus === 'loading' && 'Loading model…'}
            {modelStatus === 'ready' && 'Language model ready'}
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
            <div className="model-notice" role="status" aria-live="polite"><span className="pixel-loader" />Loading the language model…</div>
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

          <form className="feedback-form" onSubmit={submitFeedback}>
            <label htmlFor="feedback-input">Feedback (up to 500 characters)</label>
            <textarea
              id="feedback-input"
              maxLength={500}
              value={feedbackText}
              onChange={(event) => setFeedbackText(event.target.value)}
              placeholder="Example: That last move was bad. Please take a dish instead."
              disabled={game.status === 'waiting' || modelStatus !== 'ready'}
            />
            <div><span>{feedbackText.length}/500</span><button disabled={!feedbackText.trim() || processingFeedback || game.status === 'waiting'}>{processingFeedback ? 'Analyzing…' : 'Submit feedback'}</button></div>
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
              {visibleFeedback.error && (
                <div className="model-notice notice-error" role="alert">{visibleFeedback.error}</div>
              )}
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
                        ? lowConfidenceCopy(phrase.model.threshold)
                        : `Calibrated confidence · threshold ${(phrase.model.threshold * 100).toFixed(0)}%`}
                    </p>
                  </article>
                );
              })}
            </div>
          )}
        </aside>
      </section>

      {!consented && (
        <div className="consent-dialog-backdrop">
          <section
            aria-describedby="research-consent-description"
            aria-labelledby="research-consent-title"
            aria-modal="true"
            className="consent-dialog"
            role="dialog"
          >
            <span className="dialog-kicker">BEFORE YOUR SHIFT</span>
            <h2 id="research-consent-title">Research data consent</h2>
            <p id="research-consent-description">
              If you continue, this study stores a random participant ID, game actions, the feedback text you type, and language-model classifications to improve the models. Do not enter personal information. Close the page to stop future collection.
            </p>
            <label className="consent-choice">
              <input
                autoFocus
                checked={consentChecked}
                onChange={(event) => setConsentChecked(event.target.checked)}
                type="checkbox"
              />
              <span>I understand and voluntarily agree to this research data collection.</span>
            </label>
            <button
              className="primary-button"
              disabled={!consentChecked || modelStatus !== 'ready'}
              onClick={acceptConsentAndStart}
            >
              {modelStatus === 'loading' ? 'Loading…' : 'Agree and start'}
            </button>
            {modelStatus === 'error' && (
              <button className="text-button" onClick={() => window.location.reload()}>
                Retry model loading
              </button>
            )}
          </section>
        </div>
      )}

      <footer>
        <span>DURF Kitchen Lab · Research Beta</span>
        <span>Research data is collected only after consent</span>
      </footer>
    </main>
  );
}
