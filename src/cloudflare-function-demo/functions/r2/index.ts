import { Env, KEY } from './shared';

const html = `
<form method="post" enctype="multipart/form-data" action="r2/upload">
  <input type="file" name="${KEY}">
  <br><br>
  <input disabled type="submit" value="Upload">
<form>
<script>
  document.querySelector('input[type="file"]').addEventListener('change', () => {
    document.querySelector('input[type="submit"]').disabled = false;
  });
</script>

<br><br>
<a href="r2/file" download>
  <img src="r2/file">
</a>
`;

/**
 * @see {@link https://developers.cloudflare.com/pages/platform/functions/bindings/#r2-buckets}
 */
export const onRequestGet: PagesFunction<Env> = async (context) => {
  return new Response(html, {
    headers: { 'content-type': 'text/html' },
  });
};
