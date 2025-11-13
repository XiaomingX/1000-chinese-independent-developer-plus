const html = `
<form>
  <input name="key" value="value">
  <input type="submit" value="POST">
</form>
<script>
  document.querySelector('form').addEventListener('submit', (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);
    fetch('', {
      method: 'POST',
      headers: {
      'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        key: formData.get('key'),
      }),
    });
  });
</script>
`;

export const onRequest: PagesFunction = async (context) => {
  if (context.request.method === 'GET') {
    return new Response(html, {
      headers: { 'content-type': 'text/html' },
    });
  }

  if (context.request.method === 'POST') {
    const body = await context.request.json();
    return new Response(JSON.stringify(body, null, 2), {
      headers: { 'content-type': 'application/json' },
    });
  }
};
