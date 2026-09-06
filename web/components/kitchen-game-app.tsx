'use client';

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from 'react';
import {
  applyRoute2Gaussian,
  createFullGaussianPrior,
  createIndependentGaussianPrior,
  type FeedbackFormPrediction,
  type FullGaussianState,
  type IndependentGaussianState,
} from '@/lib/browser-models';
import { loadSyntheticGameModels, type GameModels } from '@/lib/synthetic-browser-models';
import { groundRoute1Feedback } from '@/lib/route1-grounding';
import { buildGroundingResearchTrace } from '@/lib/grounding-research-trace';
import { applyGroundedPragmaticUtterance, type GroundedPragmaticRoute1Result } from '@/lib/pragmatic-route1';
import { scoreVaderSentiment } from '@/lib/vader-sentiment';
import { researchPayloadFits } from '@/lib/research-limits';
import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  GAME_STEP_INTERVAL_MS,
  STEPS_PER_SECOND,
  STATIONS,
  TERRAIN_ROWS,
  chooseAiDecision,
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
import type { RewardWeights } from '@/lib/subgoal-policy';
import {
  TICK_SUMMARY_RECEIPT_SCHEMA_VERSION,
  aiExecutionResult,
  counterfactualBehaviorChanged,
  type AiTickReceipt,
  type PolicyExecutionReceipt,
} from '@/lib/policy-execution-receipt';
import { configuredFeedbackRoute } from '@/lib/feedback-route-config';
import {
  buildFeedbackRouteTrace,
  gameForPolicyProposal,
} from '@/lib/feedback-policy-contract';
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
  FEEDBACK_SEMANTICS_VERSION,
  type PhraseResearchPrediction,
} from '@/lib/research-types';
import {
  aggregateProbabilities,
  gameFeatureCounts,
  splitFeedbackPhrases,
  toResearchPrediction,
  topResearchLabel,
} from '@/lib/route-inputs';
import {
  appendTrajectoryStep,
  featurizeTrajectorySteps,
  normalizeTrajectoryFeatures,
  recordTrajectoryStep,
  type TrajectoryStep,
} from '@/lib/trajectory-featurizer';

interface VisiblePhrase {
  model: FeedbackFormPrediction;
  research: PhraseResearchPrediction;
}

interface VisibleFeedback {
  phrases: VisiblePhrase[];
  outcome?: string;
  policyImpact?: PolicyImpact;
  error?: string;
}

interface RewardPolicySnapshot {
  weights: RewardWeights;
  revision: number;
  sourceFeedbackId: string | null;
  modelHash: string | null;
}

interface PolicyImpact {
  feedbackId: string;
  roundId: string;
  applied: boolean;
  revision: number;
  evaluatedAtTick: number;
  beforeSubgoal: string;
  beforeAction: GameAction;
  afterSubgoal: string;
  afterAction: GameAction;
  planChanged: boolean;
  executionReceipt?: PolicyExecutionReceipt;
}

interface PendingPolicyReceipt {
  feedbackId: string;
  roundId: string;
  revision: number;
  previousWeights: RewardWeights;
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

function posteriorWeights(features: string[], mean: number[]): RewardWeights {
  if (features.length !== mean.length) {
    throw new Error('posterior feature schema does not match its mean');
  }
  return Object.fromEntries(
    features.map((feature, index) => [feature, mean[index]]),
  ) as RewardWeights;
}

function subgoalCopy(subgoal: string): string {
  return subgoal.toLowerCase().replaceAll('_', ' ');
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
                <span aria-hidden="true" className="chef-body" />
                <span aria-hidden="true" className="chef-face" />
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
              aria-label={`${LABEL_COPY[label].name} score ${(score * 100).toFixed(1)}%`}
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
  const [models, setModels] = useState<GameModels | null>(null);
  const [modelStatus, setModelStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [feedbackText, setFeedbackText] = useState('');
  const [visibleFeedback, setVisibleFeedback] = useState<VisibleFeedback | null>(null);
  const [processingFeedback, setProcessingFeedback] = useState(false);
  const queueRef = useRef<ResearchEventQueue | null>(null);
  const route1Ref = useRef<FullGaussianState | null>(null);
  const route2Ref = useRef<IndependentGaussianState | null>(null);
  const policyRef = useRef<RewardPolicySnapshot>({
    weights: {},
    revision: 0,
    sourceFeedbackId: null,
    modelHash: null,
  });
  const roundIdRef = useRef<string | null>(null);
  const roundEpochRef = useRef(0);
  const trajectoryRef = useRef<TrajectoryStep[]>([]);
  const processingFeedbackRef = useRef(false);
  const pendingPolicyReceiptRef = useRef<PendingPolicyReceipt | null>(null);
  const [awaitingPolicyReceipt, setAwaitingPolicyReceipt] = useState(false);
  const pendingHumanActionRef = useRef<GameAction>('stay');
  const pressedHumanMovementKeysRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    let cancelled = false;
    const requested = new URLSearchParams(window.location.search).getAll('classifier');
    if (requested.length > 1 || (requested.length === 1 && requested[0] !== 'synthetic-v1')) {
      setModelStatus('error');
      return () => {
        cancelled = true;
      };
    }
    loadSyntheticGameModels()
      .then((loaded) => {
        if (cancelled) return;
        setModels(loaded);
        route1Ref.current = createFullGaussianPrior(loaded.features);
        route2Ref.current = createIndependentGaussianPrior(loaded.features);
        setModelStatus('ready');
      })
      .catch(() => {
        if (cancelled) return;
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
    if (processingFeedbackRef.current) return;
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
      const activePolicy = policyRef.current;
      const aiDecision = chooseAiDecision(previous, activePolicy.weights);
      const aiAction = aiDecision.action;
      const next = stepGame(previous, aiAction, humanAction);
      const roundId = roundIdRef.current;
      if (!roundId) return;
      const execution = aiExecutionResult(previous, next, aiAction);
      const aiReceipt: AiTickReceipt = {
        roundId,
        policyRevision: activePolicy.revision,
        evaluatedAtTick: previous.tick,
        completedAtTick: next.tick,
        proposedSubgoal: aiDecision.chosenSubgoal,
        proposedAction: aiAction,
        humanAction,
        ...execution,
      };
      const pendingReceipt = pendingPolicyReceiptRef.current;
      let policyExecutionReceipt: PolicyExecutionReceipt | null = null;
      if (
        pendingReceipt &&
        pendingReceipt.roundId === roundId &&
        pendingReceipt.revision === activePolicy.revision
      ) {
        const counterfactual = chooseAiDecision(previous, pendingReceipt.previousWeights);
        const counterfactualNext = stepGame(
          previous,
          counterfactual.action,
          humanAction,
        );
        const counterfactualComparison = counterfactualBehaviorChanged(
          next,
          counterfactualNext,
        );
        policyExecutionReceipt = {
          ...aiReceipt,
          feedbackId: pendingReceipt.feedbackId,
          counterfactualSubgoal: counterfactual.chosenSubgoal,
          counterfactualAction: counterfactual.action,
          counterfactualContract: 'same_previous_state_same_human_action',
          ...counterfactualComparison,
        };
      }
      const nextTrajectory = appendTrajectoryStep(
        trajectoryRef.current,
        recordTrajectoryStep(previous, next, aiAction, humanAction),
      );
      const tickSummaryPayload = {
        receiptSchemaVersion: TICK_SUMMARY_RECEIPT_SCHEMA_VERSION,
        tick: next.tick,
        roundId,
        requestedJointActions: { ai: aiAction, human: humanAction },
        aiExecutionReceipt: aiReceipt,
        ...(policyExecutionReceipt ? { policyExecutionReceipt } : {}),
        stateBefore: gameStateSnapshot(previous),
        stateAfter: gameStateSnapshot(next),
        reward: next.score - previous.score,
        done: next.status === 'finished',
        stepEvents: next.lastStepEvents,
        featureCounts: gameFeatureCounts(next),
        recentTrajectoryFeatures: normalizeTrajectoryFeatures(
          featurizeTrajectorySteps(nextTrajectory),
        ),
        learnedPolicy: {
          revision: activePolicy.revision,
          sourceFeedbackId: activePolicy.sourceFeedbackId,
          modelHash: activePolicy.modelHash,
          chosenSubgoal: aiDecision.chosenSubgoal,
          evaluatedAtTick: previous.tick,
          proposedAction: aiAction,
          actuallyExecuted: execution.actuallyExecuted,
          executionOutcome: execution.executionOutcome,
          decisionSource: aiDecision.decisionSource,
          rewardMargin: aiDecision.rewardMargin,
          candidateScores: aiDecision.ranking.map(({ subgoal, score }) => ({
            subgoal,
            score,
          })),
          topContributions: aiDecision.topContributions,
        },
      };
      queueRef.current?.enqueue('tick_summary', tickSummaryPayload);

      trajectoryRef.current = nextTrajectory;
      gameRef.current = next;
      setGame(next);
      if (policyExecutionReceipt && pendingReceipt) {
        pendingPolicyReceiptRef.current = null;
        setAwaitingPolicyReceipt(false);
        setVisibleFeedback((current) =>
          current?.policyImpact?.feedbackId === pendingReceipt.feedbackId
            ? {
                ...current,
                policyImpact: {
                  ...current.policyImpact,
                  executionReceipt: policyExecutionReceipt ?? undefined,
                },
              }
            : current,
        );
      }
    }, GAME_STEP_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [game.status]);

  const startRound = useCallback((consentedAt?: number) => {
    if (!models || modelStatus !== 'ready') return;
    if (processingFeedbackRef.current) return;
    const previousRoundId = roundIdRef.current;
    const cancelledReceipt = pendingPolicyReceiptRef.current;
    const nextRoundId = crypto.randomUUID();
    if (!queueRef.current) {
      if (consentedAt === undefined) return;
      queueRef.current = new ResearchEventQueue(
        getOrCreateAnonymousUserId(),
        undefined,
        undefined,
        consentedAt,
      );
    } else {
      queueRef.current.enqueue('restart', {
        ...gameSummary(gameRef.current),
        previousRoundId,
        nextRoundId,
        resetPolicyRevision: policyRef.current.revision,
        ...(cancelledReceipt
          ? {
              cancelledPolicyReceipt: {
                receiptSchemaVersion: TICK_SUMMARY_RECEIPT_SCHEMA_VERSION,
                feedbackId: cancelledReceipt.feedbackId,
                roundId: cancelledReceipt.roundId,
                policyRevision: cancelledReceipt.revision,
                reason: 'round_restarted_before_next_tick',
              },
            }
          : {}),
      });
    }
    roundEpochRef.current += 1;
    roundIdRef.current = nextRoundId;
    route1Ref.current = createFullGaussianPrior(models.features);
    route2Ref.current = createIndependentGaussianPrior(models.features);
    policyRef.current = {
      weights: {},
      revision: 0,
      sourceFeedbackId: null,
      modelHash: null,
    };
    trajectoryRef.current = [];
    pendingPolicyReceiptRef.current = null;
    pendingHumanActionRef.current = 'stay';
    pressedHumanMovementKeysRef.current.clear();
    const next = startGame(createGameState());
    gameRef.current = next;
    setGame(next);
    setVisibleFeedback(null);
    setAwaitingPolicyReceipt(false);
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
    const stateAtSubmission = gameRef.current;
    const roundId = roundIdRef.current;
    if (
      !utterance ||
      !models ||
      !roundId ||
      !['running', 'paused'].includes(stateAtSubmission.status) ||
      processingFeedbackRef.current ||
      pendingPolicyReceiptRef.current
    ) {
      return;
    }
    const phrases = splitFeedbackPhrases(utterance);
    if (phrases.length === 0) return;
    const submissionSnapshot = {
      roundId,
      roundEpoch: roundEpochRef.current,
      tick: stateAtSubmission.tick,
      game: stateAtSubmission,
      trajectory: [...trajectoryRef.current],
      policy: {
        ...policyRef.current,
        weights: { ...policyRef.current.weights },
      },
      route1: route1Ref.current,
      route2: route2Ref.current,
    };
    processingFeedbackRef.current = true;
    setProcessingFeedback(true);

    try {
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      if (
        submissionSnapshot.roundEpoch !== roundEpochRef.current ||
        submissionSnapshot.roundId !== roundIdRef.current
      ) {
        return;
      }
      if (gameRef.current.status === 'finished') {
        setVisibleFeedback({
          phrases: [],
          error: 'The round ended before the feedback was ready, so it was not applied.',
        });
        return;
      }
      const modelPredictions = phrases.map((phrase) => models.classify(phrase));
      const visiblePhrases = modelPredictions.map((prediction, index) => ({
        model: prediction,
        research: toResearchPrediction(phrases[index], prediction),
      }));
      const aggregate = aggregateProbabilities(modelPredictions);
      const lowConfidence = modelPredictions.some((prediction) => prediction.abstained);
      const classifierHash = modelPredictions[0].modelHash;
      const feedbackId = crypto.randomUUID();
      const policyBefore = submissionSnapshot.policy;
      const proposalGame = gameForPolicyProposal(submissionSnapshot.game);
      const beforeDecision = chooseAiDecision(
        proposalGame,
        policyBefore.weights,
      );
      const trajectoryFeatures = normalizeTrajectoryFeatures(
        featurizeTrajectorySteps(submissionSnapshot.trajectory),
      );
      let outcome = '';
      let traceResult: 'updated' | 'rejected' = 'updated';
      let traceReason: string | undefined;
      let changedFeatures: Array<[string, number]> = [];
      let updaterModelHash = classifierHash;
      let nextPosteriorMean: number[] | null = null;
      let nextRoute1State = submissionSnapshot.route1;
      let nextRoute2State = submissionSnapshot.route2;
      const groundingResults: GroundedPragmaticRoute1Result[] = [];

      if (DEPLOYMENT_FEEDBACK_ROUTE === 'route1') {
        const prior = submissionSnapshot.route1 ?? createFullGaussianPrior(models.features);
        updaterModelHash = models.groundingModelHash;
        const observations = phrases.map((phraseText) => {
          const reference = models.ground(phraseText);
          const grounding = groundRoute1Feedback({
            text: phraseText, grounding: reference, state: proposalGame, trajectoryFeatures,
          });
          return {
            grounding, sentiment: scoreVaderSentiment(phraseText).compound, basePrecision: 2,
          };
        });
        const transaction = applyGroundedPragmaticUtterance(prior, observations);
        groundingResults.push(...transaction.results);
        const candidate = transaction.state;
        const rejected = transaction.status === 'rejected';
        if (!rejected) {
          nextRoute1State = candidate;
          nextPosteriorMean = candidate.mean;
        }
        const mergedDelta = rejected
          ? {}
          : Object.fromEntries(
              models.features.map((feature, index) => [
                feature,
                candidate.mean[index] - prior.mean[index],
              ]),
            );
        changedFeatures = mostChanged(mergedDelta);
        const reason = transaction.reason;
        traceResult = rejected ? 'rejected' : 'updated';
        traceReason = rejected ? reason ?? 'grounding_rejected' : undefined;
        outcome = rejected
          ? `Route 1 did not update the weights: ${reason ?? 'grounding_rejected'}`
          : `Route 1 applied ${groundingResults.length} independently grounded pragmatic updates`;
      } else {
        const prior = submissionSnapshot.route2 ?? createIndependentGaussianPrior(models.features);
        const prediction = models.route2(utterance, trajectoryFeatures);
        const result = applyRoute2Gaussian(prior, prediction.weights, 2);
        nextRoute2State = result.state;
        nextPosteriorMean = result.state.mean;
        updaterModelHash = prediction.modelHash;
        changedFeatures = mostChanged(result.delta);
        outcome = `Route 2 used ${prediction.ensembleSize} models to update all ${models.features.length} reward weights`;
      }
      const nextPolicy: RewardPolicySnapshot = nextPosteriorMean
        ? {
            weights: posteriorWeights(models.features, nextPosteriorMean),
            revision: policyBefore.revision + 1,
            sourceFeedbackId: feedbackId,
            modelHash: updaterModelHash,
          }
        : policyBefore;
      const afterDecision = chooseAiDecision(
        proposalGame,
        nextPolicy.weights,
      );
      const policyImpact: PolicyImpact = {
        feedbackId,
        roundId: submissionSnapshot.roundId,
        applied: nextPolicy.revision !== policyBefore.revision,
        revision: nextPolicy.revision,
        evaluatedAtTick: submissionSnapshot.tick,
        beforeSubgoal: beforeDecision.chosenSubgoal,
        beforeAction: beforeDecision.action,
        afterSubgoal: afterDecision.chosenSubgoal,
        afterAction: afterDecision.action,
        planChanged:
          beforeDecision.chosenSubgoal !== afterDecision.chosenSubgoal ||
          beforeDecision.action !== afterDecision.action,
      };
      const nextPendingReceipt: PendingPolicyReceipt | null = policyImpact.applied
        ? {
          feedbackId,
          roundId: submissionSnapshot.roundId,
          revision: nextPolicy.revision,
          previousWeights: { ...policyBefore.weights },
        }
        : null;
      const trace = buildFeedbackRouteTrace({
        route: DEPLOYMENT_FEEDBACK_ROUTE,
        result: traceResult,
        reason: traceReason,
        classifierVariant: models.classifierVariant,
        policyRevision: nextPolicy.revision,
        snapshotStatus: submissionSnapshot.game.status,
      });
      const feedbackPayload = {
        roundId: submissionSnapshot.roundId,
        submittedAtTick: submissionSnapshot.tick,
        utterance,
        selectedRoute: DEPLOYMENT_FEEDBACK_ROUTE,
        feedbackSemanticsVersion: FEEDBACK_SEMANTICS_VERSION,
        trainingScope: 'synthetic_only',
        groundingModelHash: models.groundingModelHash,
        ...buildGroundingResearchTrace(groundingResults, phrases, traceResult === 'updated'),
        classifierModelHash: classifierHash,
        updaterModelHash,
        calibrated: modelPredictions.every((prediction) => prediction.calibrated),
        temperatureScaled: modelPredictions.every(
          (prediction) => prediction.temperatureScaled,
        ),
        independentlyCalibrated: modelPredictions.every(
          (prediction) => prediction.independentlyCalibrated,
        ),
        scoreKinds: modelPredictions.map((prediction) => prediction.scoreKind),
        scorePolicyVersion: 'synthetic-raw-softmax-v1',
        thresholdPolicy: 'synthetic_dev_threshold_v1',
        diagnosticThresholds: modelPredictions.map(
          (prediction) => prediction.threshold,
        ),
        classifierVariant: models.classifierVariant,
        classifierManifestPath: models.classifierManifestPath,
        classifierArtifactPath: models.classifierArtifactPath,
        calibrationVersions: modelPredictions.map(
          (prediction) => prediction.calibrationVersion,
        ),
        outcome,
        policyImpact,
        policyRevision: nextPolicy.revision,
        proposedAiDecision: {
          evaluatedAtTick: submissionSnapshot.tick,
          snapshotStatus: submissionSnapshot.game.status,
          proposalEvaluationStatus: proposalGame.status,
          chosenSubgoal: afterDecision.chosenSubgoal,
          action: afterDecision.action,
          decisionSource: afterDecision.decisionSource,
          rewardMargin: afterDecision.rewardMargin,
          candidateScores: afterDecision.ranking.map(({ subgoal, score }) => ({
            subgoal,
            score,
          })),
          topContributions: afterDecision.topContributions,
        },
        changedFeatures: Object.fromEntries(changedFeatures),
        gameSnapshot: {
          tick: submissionSnapshot.tick,
          score: submissionSnapshot.game.score,
          features: gameFeatureCounts(submissionSnapshot.game),
          recentTrajectoryFeatures: trajectoryFeatures,
        },
      };
      if (!researchPayloadFits(feedbackPayload)) {
        setVisibleFeedback({
          phrases: visiblePhrases,
          outcome: 'Feedback not applied. Try fewer clauses in one message.',
        });
        return;
      }
      const feedbackMetadata = {
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
      };
      if (
        submissionSnapshot.roundEpoch !== roundEpochRef.current ||
        submissionSnapshot.roundId !== roundIdRef.current
      ) {
        return;
      }
      const queue = queueRef.current;
      if (!queue || queue.enqueue('feedback', feedbackPayload, feedbackMetadata) === null) {
        throw new Error('Research event queue is unavailable');
      }

      // Commit the posterior, policy, receipt lock, and UI only after the full
      // update and its research payload have been constructed and accepted.
      route1Ref.current = nextRoute1State;
      route2Ref.current = nextRoute2State;
      policyRef.current = nextPolicy;
      pendingPolicyReceiptRef.current = nextPendingReceipt;
      setAwaitingPolicyReceipt(nextPendingReceipt !== null);
      if (traceResult === 'updated') {
        trajectoryRef.current = trajectoryRef.current.filter(
          (step) => step.stateAfter.tick > submissionSnapshot.tick,
        );
      }
      setVisibleFeedback({ phrases: visiblePhrases, outcome, policyImpact });
      setFeedbackText('');
    } catch {
      setVisibleFeedback({
        phrases: [],
        error: 'We could not process that feedback. Please try again.',
      });
    } finally {
      processingFeedbackRef.current = false;
      setProcessingFeedback(false);
    }
  };

  const statusCopy = {
    waiting: 'Waiting to start',
    running: 'Shift running',
    paused: 'Paused',
    finished: 'Shift complete',
  }[game.status];
  const displayedProposalGame = gameForPolicyProposal(game);
  const displayedAiDecision = chooseAiDecision(
    displayedProposalGame,
    policyRef.current.weights,
  );

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
            {modelStatus === 'loading' && 'Preparing AI…'}
            {modelStatus === 'ready' && 'AI ready'}
            {modelStatus === 'error' && 'AI unavailable'}
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
            <div>
              <span>AI partner's next move</span>
              <strong>
                 {subgoalCopy(displayedAiDecision.chosenSubgoal)}
                 {game.status === 'paused' ? ' · starts after resume' : ''}
              </strong>
            </div>
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
              <button
                onClick={handlePause}
                disabled={
                  !['running', 'paused'].includes(game.status) || processingFeedback
                }
              >
                {game.status === 'paused' ? 'Resume' : 'Pause'}
              </button>
              <button
                onClick={beginRound}
                disabled={
                  !consented ||
                  modelStatus !== 'ready' ||
                  processingFeedback
                }
              >
                Restart
              </button>
            </div>
          </div>

          {modelStatus === 'loading' && (
            <div className="model-notice" role="status" aria-live="polite"><span className="pixel-loader" />Preparing your AI partner…</div>
          )}
          {modelStatus === 'error' && (
            <div className="model-notice notice-error" role="alert">The AI partner could not start. Please reload the page.</div>
          )}

          <KitchenBoard game={game} />
          {game.status === 'finished' && (
            <div className="round-result" role="status">
              <span>SHIFT COMPLETE</span>
              <strong>{game.score} points</strong>
              <p>{game.ordersCompleted} order{game.ordersCompleted === 1 ? '' : 's'} completed</p>
              <button
                onClick={beginRound}
                disabled={processingFeedback}
              >
                Play again
              </button>
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
            <span>English feedback · trained on synthetic examples</span>
          </div>

          <form className="feedback-form" onSubmit={submitFeedback}>
            <label htmlFor="feedback-input">Feedback (up to 500 characters)</label>
            <textarea
              id="feedback-input"
              maxLength={500}
              value={feedbackText}
              onChange={(event) => setFeedbackText(event.target.value)}
              placeholder="Example: That last move was bad. Please take a dish instead."
              disabled={
                !['running', 'paused'].includes(game.status) ||
                modelStatus !== 'ready' ||
                processingFeedback ||
                awaitingPolicyReceipt
              }
            />
            <div>
              <span>{feedbackText.length}/500</span>
              <button
                disabled={
                  !feedbackText.trim() ||
                  processingFeedback ||
                  awaitingPolicyReceipt ||
                  !['running', 'paused'].includes(game.status)
                }
              >
                {processingFeedback
                  ? 'Analyzing…'
                  : awaitingPolicyReceipt
                    ? 'Waiting for the AI move…'
                    : 'Submit feedback'}
              </button>
            </div>
          </form>

          <div className="classifier-key">
            {(Object.keys(LABEL_COPY) as Array<keyof typeof LABEL_COPY>).map((label) => (
              <span key={label}><i style={{ background: LABEL_COPY[label].color }} />{LABEL_COPY[label].name}</span>
            ))}
          </div>

          {!visibleFeedback && (
            <div className="empty-feedback">
              <div className="speech-pixels" aria-hidden="true"><i /><i /><i /></div>
              <h3>Your feedback will appear here</h3>
            </div>
          )}

          {visibleFeedback && (
            <div className="feedback-results" role="status" aria-live="polite">
              {visibleFeedback.error && (
                <div className="model-notice notice-error" role="alert">{visibleFeedback.error}</div>
              )}
              {visibleFeedback.policyImpact && (
                <div
                  className={`model-notice feedback-status ${visibleFeedback.policyImpact.applied ? 'feedback-accepted' : 'feedback-rejected'}`}
                  data-testid="feedback-status"
                >
                  <strong>
                    {visibleFeedback.policyImpact.applied
                      ? 'Feedback accepted'
                      : 'Feedback not applied'}
                  </strong>
                  <span>
                    {visibleFeedback.policyImpact.applied
                      ? visibleFeedback.policyImpact.executionReceipt
                        ? visibleFeedback.policyImpact.executionReceipt.behaviorChangeConfirmed
                          ? 'The AI used it and changed its next move.'
                          : 'The AI used it; its next move stayed the same.'
                        : game.status === 'paused'
                          ? 'The AI will use it after you resume.'
                          : 'The AI will use it on its next move.'
                      : 'The AI could not use this feedback. Try a clearer phrase.'}
                  </span>
                </div>
              )}
              {visibleFeedback.phrases.map((phrase, index) => {
                const copy = LABEL_COPY[phrase.research.label];
                const uncertain = phrase.model.abstained;
                return (
                  <article className={`phrase-card ${phrase.model.abstained ? 'low-confidence' : ''}`} key={`${phrase.research.phrase}-${index}`}>
                    <div className="phrase-number">Phrase {index + 1}</div>
                    <blockquote>{phrase.research.phrase}</blockquote>
                    <div className="prediction-line">
                      <span style={{ color: copy.color }}>
                        {copy.name}{uncertain ? ' · uncertain' : ''}
                      </span>
                      <strong>{(phrase.model.confidence * 100).toFixed(1)}%</strong>
                    </div>
                    <ProbabilityRows phrase={phrase} />
                    <p className="confidence-warning">Raw model scores · not measured accuracy</p>
                    {uncertain && (
                      <p className="confidence-warning">Not sure — try a clearer sentence.</p>
                    )}
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
                Reload and try again
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
