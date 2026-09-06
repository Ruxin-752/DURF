import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import ts from 'typescript';
import { createFullGaussianPrior } from '../lib/browser-models';
import { chooseAiDecision, createGameState, stepGame } from '../lib/game';
import { aiExecutionResult, counterfactualBehaviorChanged } from '../lib/policy-execution-receipt';
import { applyGroundedPragmaticUtterance } from '../lib/pragmatic-route1';
import { groundRoute1Feedback } from '../lib/route1-grounding';
import { REWARD_FEATURES, type RewardWeights } from '../lib/subgoal-policy';
import { scoreVaderSentiment } from '../lib/vader-sentiment';

const initial = createGameState('running');
const game = { ...initial, partner: { ...initial.partner, x: 1, y: 4, facing: 'left' as const } };
const observation = (text: string) => ({
  grounding: groundRoute1Feedback({ text, grounding: { label: 'action', confidence: 0.9 }, state: game }),
  sentiment: scoreVaderSentiment(text).compound,
});
const component = readFileSync(join(import.meta.dirname, '..', 'components', 'kitchen-game-app.tsx'), 'utf8');
const componentAst = ts.createSourceFile('kitchen-game-app.tsx', component, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);

function findNodes(node: ts.Node, predicate: (candidate: ts.Node) => boolean): ts.Node[] {
  const matches: ts.Node[] = [];
  const visit = (candidate: ts.Node) => {
    if (predicate(candidate)) matches.push(candidate);
    ts.forEachChild(candidate, visit);
  };
  visit(node);
  return matches;
}

function submissionTry(): ts.TryStatement {
  const declaration = findNodes(componentAst, (node) =>
    ts.isVariableDeclaration(node) && node.name.getText(componentAst) === 'submitFeedback')[0] as ts.VariableDeclaration;
  const handler = declaration.initializer as ts.ArrowFunction;
  const block = handler.body as ts.Block;
  const statement = block.statements.find(ts.isTryStatement);
  expect(statement).toBeDefined();
  return statement!;
}

describe('grounded utterance transaction integration', () => {
  it.each([
    ['Pick an onion.', 'Serve the soup.'],
    ['Serve the soup.', 'Pick an onion.'],
  ])('rolls back every phrase when any reference is infeasible: %j', (...phrases) => {
    const prior = createFullGaussianPrior([...REWARD_FEATURES]);
    const snapshot = structuredClone(prior);
    const result = applyGroundedPragmaticUtterance(prior, phrases.map(observation));
    expect(result).toMatchObject({ status: 'rejected', reason: 'infeasible_action_reference' });
    expect(result.state).toBe(prior);
    expect(prior).toEqual(snapshot);
    expect(result.results.some((item) => item.status === 'updated')).toBe(true);
    expect(chooseAiDecision(game, Object.fromEntries(result.state.features.map((feature, index) => [feature, result.state.mean[index]])) as RewardWeights)).toMatchObject({ chosenSubgoal: 'GET_TOMATO', action: 'interact' });
  });

  it('commits accepted feedback and verifies execution against the same-state counterfactual', () => {
    const prior = createFullGaussianPrior([...REWARD_FEATURES]);
    const result = applyGroundedPragmaticUtterance(prior, [observation('Pick an onion.')]);
    const weights = Object.fromEntries(result.state.features.map((feature, index) => [feature, result.state.mean[index]])) as RewardWeights;
    const before = chooseAiDecision(game);
    const after = chooseAiDecision(game, weights);
    const actual = stepGame(game, after.action, 'stay');
    const counterfactual = stepGame(game, before.action, 'stay');
    expect(result.status).toBe('updated');
    expect(aiExecutionResult(game, actual, after.action)).toEqual({ actuallyExecuted: true, executionOutcome: 'moved' });
    expect(counterfactualBehaviorChanged(actual, counterfactual).behaviorChangeConfirmed).toBe(true);
  });

  it('rejects an empty utterance without fabricating an applied revision', () => {
    const prior = createFullGaussianPrior([...REWARD_FEATURES]);
    expect(applyGroundedPragmaticUtterance(prior, [])).toEqual({ state: prior, status: 'rejected', reason: 'empty_utterance', results: [] });
  });

  it('connects the independent transaction, retains rejected trajectory, and marks committed phrase diagnostics', () => {
    const branch = component.slice(component.indexOf("if (DEPLOYMENT_FEEDBACK_ROUTE === 'route1')"), component.indexOf('const nextPolicy: RewardPolicySnapshot'));
    expect(branch).toContain('models.ground(phraseText)');
    expect(branch).toContain('applyGroundedPragmaticUtterance(prior, observations)');
    expect(branch).not.toContain('modelPredictions');
    expect(branch).not.toContain('lowConfidence');
    expect(component).toContain("buildGroundingResearchTrace(groundingResults, phrases, traceResult === 'updated')");
    expect(component).toMatch(/if \(traceResult === 'updated'\) \{\s*trajectoryRef\.current = trajectoryRef\.current\.filter/u);
  });

  it('returns from oversized feedback before enqueue and every posterior, policy or receipt commit', () => {
    const transaction = submissionTry();
    const guard = transaction.tryBlock.statements.find((node) =>
      ts.isIfStatement(node) && node.expression.getText(componentAst) === '!researchPayloadFits(feedbackPayload)') as ts.IfStatement;
    expect(guard).toBeDefined();
    expect(ts.isBlock(guard.thenStatement)).toBe(true);
    const rejected = guard.thenStatement as ts.Block;
    expect(rejected.statements).toHaveLength(2);
    expect(rejected.statements[0].getText(componentAst)).toContain('setVisibleFeedback(');
    expect(rejected.statements[0].getText(componentAst)).toContain('Feedback not applied. Try fewer clauses in one message.');
    expect(ts.isReturnStatement(rejected.statements[1])).toBe(true);
    expect((rejected.statements[1] as ts.ReturnStatement).expression).toBeUndefined();

    const enqueues = findNodes(transaction.tryBlock, (node) => ts.isCallExpression(node) &&
      node.expression.getText(componentAst) === 'queue.enqueue');
    expect(enqueues).toHaveLength(1);
    expect(enqueues[0].getStart(componentAst)).toBeGreaterThan(guard.end);
    const committedRefs = ['route1Ref.current', 'route2Ref.current', 'policyRef.current', 'pendingPolicyReceiptRef.current'];
    for (const target of committedRefs) {
      const writes = findNodes(transaction.tryBlock, (node) => ts.isBinaryExpression(node) &&
        node.operatorToken.kind === ts.SyntaxKind.EqualsToken && node.left.getText(componentAst) === target);
      expect(writes).toHaveLength(1);
      expect(writes[0].getStart(componentAst)).toBeGreaterThan(enqueues[0].end);
    }
  });

  it('keeps input and trajectory on the early return while always releasing the processing lock', () => {
    const transaction = submissionTry();
    const guard = transaction.tryBlock.statements.find((node) =>
      ts.isIfStatement(node) && node.expression.getText(componentAst) === '!researchPayloadFits(feedbackPayload)') as ts.IfStatement;
    expect(guard).toBeDefined();
    const inputClears = findNodes(transaction.tryBlock, (node) => ts.isCallExpression(node) &&
      node.expression.getText(componentAst) === 'setFeedbackText');
    expect(inputClears).toHaveLength(1);
    expect(inputClears[0].getStart(componentAst)).toBeGreaterThan(guard.end);
    const trajectoryWrites = findNodes(transaction.tryBlock, (node) => ts.isBinaryExpression(node) &&
      node.operatorToken.kind === ts.SyntaxKind.EqualsToken && node.left.getText(componentAst) === 'trajectoryRef.current');
    expect(trajectoryWrites).toHaveLength(1);
    expect(trajectoryWrites[0].getStart(componentAst)).toBeGreaterThan(guard.end);
    expect(transaction.finallyBlock).toBeDefined();
    expect(transaction.finallyBlock!.statements.map((node) => node.getText(componentAst))).toEqual([
      'processingFeedbackRef.current = false;',
      'setProcessingFeedback(false);',
    ]);
  });
});
