const vscode = acquireVsCodeApi();

document.addEventListener('DOMContentLoaded', () => {
    const clearBtn = document.getElementById('clear-btn');
    const canvas = document.getElementById('weeklyChart');
    if (!canvas) return;

    
    const style = getComputedStyle(document.body);
    const accent = style.getPropertyValue('--accent').trim();
    const gridColor = style.getPropertyValue('--border').trim();
    const textColor = style.getPropertyValue('--fg').trim();

    const ctx = canvas.getContext('2d');
    new Chart(ctx, {
        type: 'line', 
        data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [
                {
                    label: 'Safe Snippets',
                    data: [150, 200, 180, 220, 170, 90, 100],
                    borderColor: accent,
                    backgroundColor: accent + '22',
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Threats',
                    data: [2, 50, 1, 0, 4, 1, 1],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    fill: true,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { 
                legend: { labels: { color: textColor, font: { size: 10 } } } 
            },
            scales: {
                y: { 
                    grid: { color: gridColor }, 
                    ticks: { color: textColor } 
                },
                x: { 
                    grid: { display: false }, 
                    ticks: { color: textColor } 
                }
            }
        }
    });

    clearBtn?.addEventListener('click', () => {
        vscode.postMessage({ command: 'resolveThreats' });
    });
});