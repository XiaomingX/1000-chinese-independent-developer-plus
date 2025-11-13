/**
 * @see {@link https://developers.cloudflare.com/workers/examples/return-html/}
 */

const html = `
<h1>HTML</h1>
<a href="/">Home</a>
`;

export const onRequest: PagesFunction = async (context) => {
  return new Response(html, {
    headers: { 'Content-Type': 'text/html' },
  });
};
