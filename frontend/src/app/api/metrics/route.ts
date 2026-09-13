export const dynamic = "force-dynamic";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export async function GET() {
  try {
    const backendRes = await fetch(`${API_BASE}/metrics`, { cache: "no-store" });
    const data = await backendRes.json();
    if (!backendRes.ok) {
      return Response.json({ detail: "Metrics unavailable" }, { status: backendRes.status });
    }
    return Response.json(data, { status: 200 });
  } catch (error) {
    console.error("Proxy metrics error:", error);
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
