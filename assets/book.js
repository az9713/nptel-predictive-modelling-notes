(() => {
  'use strict';
  const root = document.documentElement;
  const read = key => { try { return localStorage.getItem(key); } catch { return null; } };
  const save = (key, value) => { try { localStorage.setItem(key, value); } catch {} };
  const theme = read('predictive-theme');
  if (theme === 'light') root.dataset.theme = theme;
  let fontSize = Number(read('predictive-font')) || (innerWidth < 700 ? 17 : 18);
  const setSize = () => { root.style.setProperty('--size', `${fontSize}px`); save('predictive-font', fontSize); };
  setSize();
  document.querySelector('[data-action=theme]')?.addEventListener('click', () => {
    root.dataset.theme = root.dataset.theme === 'light' ? 'dark' : 'light';
    save('predictive-theme', root.dataset.theme);
  });
  document.querySelector('[data-action=smaller]')?.addEventListener('click', () => { fontSize = Math.max(14, fontSize - 1); setSize(); });
  document.querySelector('[data-action=larger]')?.addEventListener('click', () => { fontSize = Math.min(25, fontSize + 1); setSize(); });
  document.querySelector('[data-action=print]')?.addEventListener('click', () => window.print());
  const navigation = document.querySelector('.sidebar > details');
  if (navigation && matchMedia('(max-width:700px)').matches) navigation.open = false;
  document.querySelectorAll('pre').forEach(pre => {
    if (pre.closest('.codewrap')) return;
    const wrapper = document.createElement('div');
    wrapper.className = 'codewrap';
    pre.before(wrapper);
    wrapper.append(pre);
    const button = document.createElement('button');
    button.className = 'copybtn';
    button.type = 'button';
    button.textContent = 'Copy';
    button.setAttribute('aria-label', 'Copy code or output');
    button.addEventListener('click', async () => {
      const content = pre.querySelector('code')?.textContent ?? pre.textContent;
      try {
        let copied = false;
        if (navigator.clipboard && window.isSecureContext) {
          try {
            await navigator.clipboard.writeText(content);
            copied = true;
          } catch {}
        }
        if (!copied) {
          const area = document.createElement('textarea');
          area.value = content;
          area.style.position = 'fixed';
          area.style.opacity = '0';
          document.body.append(area);
          area.select();
          try { copied = document.execCommand('copy'); }
          finally { area.remove(); }
        }
        button.textContent = copied ? 'Copied' : 'Select and copy';
      } catch { button.textContent = 'Select and copy'; }
      setTimeout(() => { button.textContent = 'Copy'; }, 1500);
    });
    wrapper.append(button);
  });
  document.querySelectorAll('main table').forEach(table => {
    if (table.parentElement.classList.contains('table-scroll')) return;
    const wrapper = document.createElement('div');
    wrapper.className = 'table-scroll';
    wrapper.tabIndex = 0;
    wrapper.setAttribute('role', 'region');
    wrapper.setAttribute('aria-label', 'Scrollable table');
    table.before(wrapper);
    wrapper.append(table);
  });
  let previouslyClosed = [];
  addEventListener('beforeprint', () => {
    previouslyClosed = [...document.querySelectorAll('main details:not([open])')];
    previouslyClosed.forEach(item => { item.open = true; });
  });
  addEventListener('afterprint', () => previouslyClosed.forEach(item => { item.open = false; }));
  const input = document.querySelector('#course-search');
  const results = document.querySelector('#search-results');
  if (input && results) input.addEventListener('input', () => {
    const query = input.value.trim().toLowerCase();
    results.replaceChildren();
    if (query.length < 2) return;
    const terms = query.split(/\s+/);
    const matches = (window.COURSE_SEARCH || []).filter(item => terms.every(term => item.text.toLowerCase().includes(term)));
    const count = document.createElement('p');
    count.textContent = `${matches.length} matching chapter${matches.length === 1 ? '' : 's'}`;
    results.append(count);
    const list = document.createElement('ul');
    matches.forEach(item => {
      const row = document.createElement('li');
      const link = document.createElement('a');
      link.href = item.url;
      link.textContent = `${item.number}. ${item.title}`;
      const snippet = document.createElement('p');
      const index = item.text.toLowerCase().indexOf(terms[0]);
      snippet.textContent = '…' + item.text.slice(Math.max(0, index - 80), index + 180) + '…';
      row.append(link, snippet);
      list.append(row);
    });
    results.append(list);
  });
})();
