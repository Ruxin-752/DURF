import type { FeedbackRoute } from './research-types';

export function configuredFeedbackRoute(
  value: string | undefined = process.env.NEXT_PUBLIC_FEEDBACK_ROUTE,
): FeedbackRoute {
  if (value === undefined || value === '' || value === 'route1') return 'route1';
  if (value === 'route2') return 'route2';
  throw new Error('NEXT_PUBLIC_FEEDBACK_ROUTE must be route1 or route2');
}
