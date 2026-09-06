import { env } from 'cloudflare:workers';
import { handleSyntheticDiagnosticRequest } from '@/lib/research-diagnostic';

export async function POST(request: Request): Promise<Response> {
  return handleSyntheticDiagnosticRequest(request, env as unknown as Env);
}
