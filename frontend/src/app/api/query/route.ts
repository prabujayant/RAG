export const dynamic = "force-dynamic";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export async function POST(request: Request) {
  try {
    const body = await request.json();

    const backendRes = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const data = await backendRes.json();

    if (!backendRes.ok) {
      return Response.json(
        {
          detail: data?.detail ?? { message: "Query failed" },
        },
        { status: backendRes.status }
      );
    }

    return Response.json(data, { status: 200 });
  } catch (error) {
    console.error("Proxy query error:", error);
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
