export const onRequest: PagesFunction = async (context) => {
  const response = await fetch('http://example.com/');
  const html = await response.text();
  return new Response(html, {
    headers: { 'Content-Type': 'text/html' },
  });
};
