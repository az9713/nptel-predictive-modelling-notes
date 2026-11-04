window.MathJax = {
  loader: {paths: {mathjax: 'assets'}, load: ['[tex]/boldsymbol']},
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    packages: {'[+]': ['boldsymbol']},
    tags: 'ams'
  },
  svg: {fontCache: 'local'},
  options: {skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']}
};
