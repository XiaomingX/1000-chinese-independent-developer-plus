/**
 * @see {@link https://developers.cloudflare.com/workers/examples/read-post/}
 */

const getHtml = `
<h1>Form</h1>
<form method="post">
  <input name="key" value="">
  <input type="submit" value="Submit">
</form>

<h1>JSON</h1>
<form id="json">
  <input name="key" value="">
  <input type="submit" value="Submit">
</form>
<script>
  document.getElementById('json').addEventListener('submit', (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);
    fetch('', {
      method: 'post',
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

export const onRequestGet: PagesFunction = async (context) => {
  return new Response(getHtml, {
    headers: { 'content-type': 'text/html' },
  });
};

export const onRequestPost: PagesFunction = async (context) => {
  let body = '';
  const contentType = context.request.headers.get('content-type');

  switch (contentType) {
    case 'application/json':
      body = JSON.stringify(await context.request.json());
      break;

    case 'application/text':
      body = await context.request.text();
      break;

    case 'text/html':
      body = await context.request.text();
      break;

    case 'application/x-www-form-urlencoded':
      const formData = await context.request.formData();
      const data = {};
      formData.forEach((value, key) => {
        data[key] = value;
      });
      body = JSON.stringify(data);
      break;

    // Perhaps some other type of data was submitted in the form
    // like an image, or some other binary data.
    default:
      body = 'other';
      break;
  }

  return new Response(body);
};
