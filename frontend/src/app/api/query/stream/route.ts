export const dynamic = "force-dynamic";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

import { waitForBackendReady } from "@/lib/backendReady";

/**
 * SSE passthrough: POST /api/query/stream -> backend POST /query/stream.
 * Streams backend events (retrieval -> reranking -> generation -> done)
 * straight to the browser so the UI renders progress instead of spinning.
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();

    // Backend may still be cold-starting after a restart/redeploy; wait for it
    // before opening the SSE connection so the browser doesn't see an error.
    await waitForBackendReady(API_BASE);

    const backendRes = await fetch(`${API_BASE}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
    });

    if (!backendRes.ok || !backendRes.body) {
      const data = await backendRes.json().catch(() => null);
      return Response.json(
        { detail: data?.detail ?? { message: "Query stream failed" } },
        { status: backendRes.status || 502 }
      );
    }

    return new Response(backendRes.body, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (error) {
    console.error("Proxy query stream error:", error);
    return Response.json(
      {
        detail: {
          message: "Unable to reach the AskMyDocs API. Check that the backend is running on port 8000.",
        },
      },
      { status: 502 }
    );
  }
}
