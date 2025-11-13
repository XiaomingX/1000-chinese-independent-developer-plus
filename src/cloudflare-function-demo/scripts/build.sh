#!/bin/bash

OUTPUT_DIRECTORY=dist
OUTPUT_FILE=$OUTPUT_DIRECTORY/index.html

mkdir -p $OUTPUT_DIRECTORY
marked --input README.md --output $OUTPUT_FILE

# https://github.com/sindresorhus/github-markdown-css#usage
HEAD=$(echo "<meta name='viewport' content='width=device-width,initial-scale=1'>
<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css'>
<style>
.markdown-body {
  box-sizing: border-box;
  min-width: 200px;
  max-width: 980px;
  margin: 0 auto;
  padding: 45px;
}
@media (max-width: 767px) {
  .markdown-body {
    padding: 15px;
  }
}
</style>
<script src='https://unpkg.com/github-corners/dist/embed.min.js' data-href='https://github.com/remarkablemark/cloudflare-functions-demo' data-target='_blank' async defer></script>")

echo "$HEAD
<body class='markdown-body'>
$(cat $OUTPUT_FILE)
</body>
" > $OUTPUT_FILE

echo "$HEAD
<body class='markdown-body'>
<h1>Not Found</h1>
<a href='/'>Home</a>
</body>
" > $OUTPUT_DIRECTORY/404.html
