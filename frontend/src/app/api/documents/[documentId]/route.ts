export const dynamic = "force-dynamic";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

/** Proxy GET /api/documents/:id -> backend GET /documents/:id (poll ingest status). */
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ documentId: string }> }
) {
  try {
    const { documentId } = await params;
    const backendRes = await fetch(
      `${API_BASE}/documents/${encodeURIComponent(documentId)}`
    );
    const data = await backendRes.json().catch(() => null);
    return Response.json(data, { status: backendRes.status });
  } catch (error) {
    console.error("Proxy document status error:", error);
    return Response.json(
      { detail: { message: "Unable to reach the AskMyDocs API." } },
      { status: 502 }
    );
  }
}
