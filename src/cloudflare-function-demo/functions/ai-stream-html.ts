/**
 * @see {@link https://blog.cloudflare.com/workers-ai-streaming/}
 */

const ENDPOINT = '/ai-stream'

const html = `
<script>
const source = new EventSource(${JSON.stringify(ENDPOINT)});
source.onmessage = (event) => {
  if (event.data === "[DONE]") {
      source.close();
      return;
  }
  const data = JSON.parse(event.data);
  document.body.innerHTML += data.response;
}
</script>
`;

export const onRequest: PagesFunction = async (context) => {
  return new Response(html, {
    headers: { 'content-type': 'text/html' },
  });
};
