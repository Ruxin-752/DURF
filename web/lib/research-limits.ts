export const MAX_RESEARCH_REQUEST_BYTES = 256 * 1024;
export const MAX_EVENT_PAYLOAD_CHARACTERS = 16_384;

/** Same payload bound on the producer and ingestion sides. */
export function researchPayloadFits(payload: Record<string, unknown>): boolean {
  return JSON.stringify(payload).length <= MAX_EVENT_PAYLOAD_CHARACTERS;
}
