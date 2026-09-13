/**
 * Backend-cold-start resilience.
 *
 * The AskMyDocs backend can take 1-3 minutes to become ready after a Space
 * restart or redeploy (Postgres init + model warmup), while the Next.js UI
 * comes up much faster. This helper lets API proxy routes wait for the backend
 * before forwarding, so the browser does not flash "Unable to reach the
 * AskMyDocs API" during a cold start. It self-heals: once the backend is ready
 * (or the budget elapses) the original request is made.
 */

const WAIT_BUDGET_MS = 60_000; // how long to wait for the backend to come up
const POLL_EVERY_MS = 3_000; // poll interval

/**
 * Poll the backend /health endpoint until it responds OK or the timeout
 * elapses. Returns true when the backend is reachable.
 */
export async function waitForBackendReady(
  apiBase: string,
  timeoutMs: number = WAIT_BUDGET_MS,
): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${apiBase}/health`, {
        method: "GET",
        signal: AbortSignal.timeout(4000),
      });
      if (res.ok) return true;
    } catch {
      // backend not ready yet — keep polling
    }
    await new Promise((r) => setTimeout(r, POLL_EVERY_MS));
  }
  return false;
}