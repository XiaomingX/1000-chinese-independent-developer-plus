/**
 * @see {@link https://developers.cloudflare.com/workers/examples/alter-headers/}
 */
export const onRequest: PagesFunction = async (context) => {
  const response = await fetch('http://example.com');

  // Clone the response so that it's no longer immutable
  const newResponse = new Response(response.body, response);

  // Add a custom header with a value
  newResponse.headers.append(
    'x-workers-hello',
    'Hello from Cloudflare Pages Functions'
  );

  // Delete headers
  newResponse.headers.delete('x-header-to-delete');
  newResponse.headers.delete('x-header2-to-delete');

  // Adjust the value for an existing header
  newResponse.headers.set('x-header-to-change', 'NewValue');
  return newResponse;
};
