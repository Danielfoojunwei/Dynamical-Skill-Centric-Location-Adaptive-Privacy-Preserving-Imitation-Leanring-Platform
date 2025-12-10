const API_KEY = "neuracore_secret_key_123";

export const fetchWithAuth = async (url, options = {}) => {
    const headers = {
        ...options.headers,
        'X-API-Key': API_KEY,
        'Content-Type': 'application/json'
    };
    return fetch(url, { ...options, headers });
};
