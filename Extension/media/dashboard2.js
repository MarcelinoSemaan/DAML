// media/dashboard.js
document.addEventListener('DOMContentLoaded', () => {
    const clearBtn = document.getElementById('clear-btn');
    const canvas = document.getElementById('weeklyChart');
    if (!canvas) return;

    const style = getComputedStyle(document.body);
    const accent = style.getPropertyValue('--accent').trim() || '#3b82f6';
    const gridColor = style.getPropertyValue('--border').trim() || '#444444';
    const textColor = style.getPropertyValue('--fg').trim() || '#cccccc';

    const ctx = canvas.getContext('2d');
    
    // Make chart globally accessible for updates
    window.weeklyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [
                {
                    label: 'Safe Files',
                    data: [150, 200, 180, 220, 170, 90, 100],
                    borderColor: '#22c55e',
                    backgroundColor: 'rgba(34, 197, 94, 0.15)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 2
                },
                {
                    label: 'AI Threats',
                    data: [2, 5, 1, 0, 4, 1, 1],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.15)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 2
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
                        font: { size: 9, family: 'var(--vscode-font-family)' },
                        boxWidth: 8,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor: 'var(--vscode-editor-background)',
                    titleColor: textColor,
                    bodyColor: textColor,
                    borderColor: gridColor,
                    borderWidth: 1
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
            },
            animation: {
                duration: 750
            }
        }
    });

    // Clear button handler
    clearBtn?.addEventListener('click', () => {
        const threatItems = document.querySelectorAll('#detections-list li[data-threat-id]');
        const threatIds = Array.from(threatItems).map(li => li.dataset.threatId);
        
        vscode.postMessage({ 
            command: 'resolveThreats',
            threatIds: threatIds 
        });
    });

    // Detection item click handlers
    document.querySelectorAll('#detections-list li').forEach(item => {
        item.addEventListener('click', () => {
            const filePath = item.dataset.threatId;
            if (filePath) {
                vscode.postMessage({
                    command: 'openFile',
                    path: filePath
                });
            }
        });
    });
});
