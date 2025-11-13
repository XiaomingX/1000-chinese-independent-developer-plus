/**
 * @see {@link https://developers.cloudflare.com/pages/platform/functions/get-started/#create-a-function}
 */
export function onRequest(context) {
  return new Response('Hello, world!');
}
