/**
 * @see {@link https://developers.cloudflare.com/pages/platform/functions/examples/cors-headers/}
 */

// Respond to OPTIONS method
export const onRequestOptions: PagesFunction = async () => {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': '*',
      'Access-Control-Allow-Methods': 'GET,OPTIONS',
      'Access-Control-Max-Age': '86400',
    },
  });
};

// Set CORS to /cors responses
export const onRequest: PagesFunction = async (context) => {
  return new Response(JSON.stringify({ cors: true }), {
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Content-Type': 'application/json',
    },
  });
};
