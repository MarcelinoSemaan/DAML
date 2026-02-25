document.addEventListener('DOMContentLoaded', () => {
    const clearBtn = document.getElementById('clear-btn');
    const canvas = document.getElementById('weeklyChart');
    if (!canvas) return;

    const style = getComputedStyle(document.body);
    const accent = style.getPropertyValue('--accent').trim() || '#3b82f6';
    const gridColor = style.getPropertyValue('--border').trim() || '#444444';
    const textColor = style.getPropertyValue('--fg').trim() || '#cccccc';

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
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 1 
                },
                {
                    label: 'Threats',
                    data: [2, 50, 1, 0, 4, 1, 1],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.15)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: { 
                legend: { 
                    position: 'top',
                    labels: { 
                        color: textColor, 
                        font: { size: 9 }, 
                        boxWidth: 8,       
                        usePointStyle: true
                    } 
                } 
            },
            scales: {
                y: { 
                    beginAtZero: true,
                    grid: { color: gridColor }, 
                    ticks: { color: textColor, font: { size: 9 }, maxTicksLimit: 5 }
                },
                x: { 
                    grid: { display: false }, 
                    ticks: { color: textColor, font: { size: 9 } } 
                }
            }
        }
    });

    clearBtn?.addEventListener('click', () => {
        vscode.postMessage({ command: 'resolveThreats' });
    });
});