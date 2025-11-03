# Opens pre-filled Google Alerts links in your default browser.
# Run from repo root: powershell -ExecutionPolicy Bypass -File .\tools\open_alert_links.ps1

$urls = @(
  'https://www.google.com/alerts?hl=en&q=%22LFM%20lattice%20field%20model%22',
  'https://www.google.com/alerts?hl=en&q=%22Lorentzian%20Field%20Model%22',
  'https://www.google.com/alerts?hl=en&q=%22Lattice%20Field%20Model%22%20%22Lorentzian%22',
  'https://www.google.com/alerts?hl=en&q=%22lfm_equation.py%22',
  'https://www.google.com/alerts?hl=en&q=%22lfm_parallel.py%22',
  'https://www.google.com/alerts?hl=en&q=%22lfm_simulator.py%22',
  'https://www.google.com/alerts?hl=en&q=%22gpartin%22%20%22LFM%22',
  'https://www.google.com/alerts?hl=en&q=%22gpartin%22%20%22Lattice%20Field%20Model%22',
  'https://www.google.com/alerts?hl=en&q=%226agn8%22%20osf.io',
  'https://www.google.com/alerts?hl=en&q=%2217478758%22%20zenodo.org',
  'https://www.google.com/alerts?hl=en&q=%22osf.io%2F6agn8%22',
  'https://www.google.com/alerts?hl=en&q=%22zenodo.org%2Frecords%2F17478758%22'
)

foreach ($u in $urls) { Start-Process $u }

Write-Host "Opened $($urls.Count) Google Alerts links in your browser. For each page, click 'Create alert' and set delivery: weekly digest." -ForegroundColor Green
