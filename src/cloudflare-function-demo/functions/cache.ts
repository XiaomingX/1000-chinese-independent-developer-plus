/**
 * @see {@link https://developers.cloudflare.com/workers/examples/cache-api/}
 * @see {@link https://developers.cloudflare.com/workers/examples/cache-post-request/}
 * @see {@link https://developers.cloudflare.com/workers/examples/cache-using-fetch/}
 */

const requestUrl = 'http://example.com';

export const onRequestGet: PagesFunction = async (context) => {
  try {
    const body = await context.request.clone().text();

    // Hash the request body to use it as a part of the cache key
    const hash = await sha256(body);
    const cacheUrl = new URL(context.request.url);

    // Store the URL in cache by prepending the body's hash
    cacheUrl.pathname = '/posts' + cacheUrl.pathname + hash;

    // Convert to a GET to be able to cache
    const cacheKey = new Request(cacheUrl.toString(), {
      headers: context.request.headers,
      method: 'GET',
    });

    // Find the cache key in the cache
    const cache = caches.default;
    let response = await cache.match(cacheKey);

    // Otherwise, fetch response to GET request from origin
    if (response) {
      console.log('Cache hit');
      return response;
    } else {
      console.log('Cache miss');
      response = await fetch(requestUrl, {
        cf: {
          // Always cache this fetch regardless of content type
          // for a max of 5 seconds before revalidating the resource
          cacheTtl: 5,
          cacheEverything: true,
        },
      });
      // Set cache control headers to cache on browser for 25 minutes
      response.headers.set('Cache-Control', 'max-age=1500');
      context.waitUntil(cache.put(cacheKey, response.clone()));
      return response;
    }
  } catch (error) {
    return new Response('Error thrown: ' + error.message);
  }
};

async function sha256(message: string): Promise<string> {
  // encode as UTF-8
  const msgBuffer = await new TextEncoder().encode(message);

  // hash the message
  const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);

  // convert bytes to hex string
  return [...new Uint8Array(hashBuffer)]
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}
