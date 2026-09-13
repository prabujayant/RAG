export const dynamic = "force-dynamic";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

/**
 * Proxy for document upload. Forwards the multipart body to the backend's
 * POST /documents/upload. Pass ?background=true to enqueue on the Celery
 * worker and return immediately (status=processing) instead of blocking
 * on parse + embed + index (~20-60s cold).
 */
export async function POST(request: Request) {
  try {
    const incoming = await request.formData();
    const file = incoming.get("file");

    if (!(file instanceof File)) {
      return Response.json(
        { detail: { message: "No file provided." } },
        { status: 400 }
      );
    }

    const outgoing = new FormData();
    outgoing.append("file", file, file.name);

    const url = new URL(request.url);
    const background = url.searchParams.get("background") === "true";

    const backendRes = await fetch(
      `${API_BASE}/documents/upload${background ? "?background=true" : ""}`,
      {
        method: "POST",
        body: outgoing,
      }
    );

    const data = await backendRes.json().catch(() => null);

    if (!backendRes.ok) {
      return Response.json(
        { detail: data?.detail ?? { message: "Upload failed" } },
        { status: backendRes.status }
      );
    }

    return Response.json(data, { status: 200 });
  } catch (error) {
    console.error("Proxy upload error:", error);
    return Response.json(
      {
        detail: {
          message:
            "Unable to reach the AskMyDocs API. Check that the backend is running on port 8000.",
        },
      },
      { status: 502 }
    );
  }
}
