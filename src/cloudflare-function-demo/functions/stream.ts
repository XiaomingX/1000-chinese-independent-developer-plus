/**
 * @see {@link https://developers.cloudflare.com/workers/learning/using-streams/}
 */
export const onRequestGet: PagesFunction = async (context) => {
  const response = await fetch('http://example.com');
  const { readable, writable } = new TransformStream();
  response.body.pipeTo(writable);
  return new Response(readable);
};
