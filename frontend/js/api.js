export const BASE_URL = "http://127.0.0.1:5000";

export async function apiFetch(path, options = {}) {
    const response = await fetch(`${BASE_URL}${path}`, options);
    if (!response.ok) {
        throw new Error(`API error ${response.status}: ${response.statusText}`);
    }
    return response;
}
