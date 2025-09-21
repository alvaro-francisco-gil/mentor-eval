// MentorEval Leaderboard JavaScript - Static Version for GitHub Pages
class MentorEvalLeaderboard {
    constructor() {
        this.data = [];
        this.runParameters = {};
        this.currentMetric = 'pearson_correlation';
        this.currentSortBy = 'overall';
        this.datasets = ['asap', 'asap2', 'ellipse', 'mohler', 'ptasag2018', 'arasag'];
        
        this.initializeEventListeners();
        this.loadData();
    }

    initializeEventListeners() {
        // Metric selector
        document.getElementById('metricSelect').addEventListener('change', (e) => {
            this.currentMetric = e.target.value;
            this.updateLeaderboard();
        });

        // Dataset column headers
        document.querySelectorAll('.clickable').forEach(header => {
            header.addEventListener('click', (e) => {
                const dataset = e.currentTarget.dataset.dataset;
                this.currentSortBy = dataset;
                this.updateLeaderboard();
            });
        });
    }

    async loadData() {
        this.showLoading(true);
        this.hideError();

        try {
            // Load run parameters first
            await this.loadRunParameters();
            
            // Load all result files
            const resultFiles = await this.getResultFiles();
            const allData = [];

            for (const file of resultFiles) {
                try {
                    // Load from results directory directly
                    const response = await fetch(`results/${file}`);
                    if (!response.ok) continue;
                    
                    const resultData = await response.json();
                    const runData = this.processResultData(resultData, file);
                    if (runData) {
                        allData.push(runData);
                    }
                } catch (error) {
                    console.warn(`Failed to load ${file}:`, error);
                }
            }

            this.data = allData;
            this.updateLeaderboard();
            this.showLoading(false);
        } catch (error) {
            this.showError('Failed to load benchmark data. Please check that the results folder is accessible.');
            this.showLoading(false);
        }
    }

    async getResultFiles() {
        // Auto-generated list of result files
        return [
            '1_gpt-4o-mini_mentoreval-test_20250920_154100.json',
            '2_gpt-4o-mini_mentoreval-test_20250921_234926.json',
            '3_gpt-4o-mini_mentoreval-test_20250921_233111.json',
            '4_mentoreval_few_shot_3.json',
            '5_mentoreval_few_shot_5.json'
        ];
    }

    async loadRunParameters() {
        try {
            // Try to load from runs directory
            const response = await fetch('runs/1_mentoreval_guidance.json');
            if (response.ok) {
                const run1 = await response.json();
                this.runParameters[1] = run1;
            }
            
            const response2 = await fetch('runs/2_mentoreval_no_guidance.json');
            if (response2.ok) {
                const run2 = await response2.json();
                this.runParameters[2] = run2;
            }
            
            const response3 = await fetch('runs/3_mentoreval_explanation.json');
            if (response3.ok) {
                const run3 = await response3.json();
                this.runParameters[3] = run3;
            }
            
            const response5 = await fetch('runs/5_mentoreval_few_shot_5.json');
            if (response5.ok) {
                const run5 = await response5.json();
                this.runParameters[5] = run5;
            }
            
            const response11 = await fetch('runs/11_test.json');
            if (response11.ok) {
                const run11 = await response11.json();
                this.runParameters[11] = run11;
            }
        } catch (error) {
            console.warn('Failed to load run parameters:', error);
        }
    }

    processResultData(resultData, filename) {
        try {
            const runInfo = resultData.run_info;
            const metricsSummary = resultData.metrics_summary;

            if (!runInfo || !metricsSummary) {
                return null;
            }

            // Calculate overall metrics (only for full benchmark runs)
            const overallMetrics = this.calculateOverallMetrics(metricsSummary, runInfo.task_name);
            
            // Skip this run if it's not a full benchmark run
            if (!overallMetrics) {
                return null;
            }
            
            // Calculate dataset-specific metrics
            const datasetMetrics = this.calculateDatasetMetrics(metricsSummary);

            // Get run parameters
            const runParams = this.getRunParameters(runInfo.run_id);

            return {
                runId: runInfo.run_id,
                modelName: runInfo.model_name,
                taskName: runInfo.task_name,
                timestamp: runInfo.timestamp,
                parameters: runParams,
                overall: overallMetrics,
                datasets: datasetMetrics,
                filename: filename
            };
        } catch (error) {
            console.error(`Error processing ${filename}:`, error);
            return null;
        }
    }

    calculateOverallMetrics(metricsSummary, taskName) {
        // Only use aggregated metrics for full benchmark runs (task_name = "mentoreval")
        if (taskName === "mentoreval" && metricsSummary.aggregated) {
            return metricsSummary.aggregated;
        }
        
        // For specific dataset tasks, return null (should not appear in overall leaderboard)
        return null;
    }

    calculateDatasetMetrics(metricsSummary) {
        const datasetMetrics = {};

        // Initialize dataset metrics
        this.datasets.forEach(dataset => {
            datasetMetrics[dataset] = {};
            const metricNames = ['exact_grade_match', 'grade_mae', 'grade_rmse', 'pearson_correlation', 'spearman_correlation', 'ks_statistic', 'wasserstein_distance'];
            metricNames.forEach(metric => {
                datasetMetrics[dataset][metric] = { value: 0, stderr: 0 };
            });
        });

        // Group tasks by dataset
        const datasetTasks = {};
        this.datasets.forEach(dataset => {
            datasetTasks[dataset] = [];
        });

        Object.keys(metricsSummary).forEach(taskName => {
            this.datasets.forEach(dataset => {
                if (taskName.includes(`mentoreval_${dataset}_`)) {
                    datasetTasks[dataset].push(metricsSummary[taskName]);
                }
            });
        });

        // Calculate dataset averages
        this.datasets.forEach(dataset => {
            const tasks = datasetTasks[dataset];
            if (tasks.length > 0) {
                const metricNames = ['exact_grade_match', 'grade_mae', 'grade_rmse', 'pearson_correlation', 'spearman_correlation', 'ks_statistic', 'wasserstein_distance'];
                
                metricNames.forEach(metric => {
                    let sum = 0;
                    let stderrSum = 0;
                    let count = 0;

                    tasks.forEach(task => {
                        if (task[metric]) {
                            sum += task[metric].value;
                            stderrSum += task[metric].stderr;
                            count++;
                        }
                    });

                    if (count > 0) {
                        datasetMetrics[dataset][metric].value = sum / count;
                        datasetMetrics[dataset][metric].stderr = stderrSum / count;
                    }
                });
            }
        });

        return datasetMetrics;
    }

    getRunParameters(runId) {
        // Try to load from API data first
        if (this.runParameters[runId]) {
            return this.runParameters[runId].parameters || this.runParameters[runId];
        }

        // Fallback to default parameters based on run ID
        const defaultParams = {
            model_name: 'gpt-4o-mini',
            training_examples: 0,
            test_samples: 5,
            task_name: 'mentoreval',
            show_guidance: true,
            explanation: false,
            show_isced_level: true
        };

        // Customize based on run ID
        switch (runId) {
            case 1:
                return { ...defaultParams, show_guidance: true, explanation: false };
            case 2:
                return { ...defaultParams, show_guidance: false, explanation: false };
            case 3:
                return { ...defaultParams, show_guidance: true, explanation: true };
            case 4:
                return { ...defaultParams, show_guidance: true, explanation: true, training_examples: 3 };
            case 5:
                return { ...defaultParams, show_guidance: true, explanation: true, training_examples: 5 };
            case 11:
                return { ...defaultParams, show_guidance: true, explanation: false };
            default:
                return defaultParams;
        }
    }

    updateLeaderboard() {
        if (this.data.length === 0) return;

        // Sort data based on current metric and sort by
        const sortedData = this.sortData();
        
        // Update table headers
        this.updateTableHeaders();
        
        // Populate table body
        this.populateTableBody(sortedData);
    }

    sortData() {
        return [...this.data].sort((a, b) => {
            let scoreA, scoreB;

            if (this.currentSortBy === 'overall') {
                scoreA = a.overall[this.currentMetric]?.value || 0;
                scoreB = b.overall[this.currentMetric]?.value || 0;
            } else {
                scoreA = a.datasets[this.currentSortBy]?.[this.currentMetric]?.value || 0;
                scoreB = b.datasets[this.currentSortBy]?.[this.currentMetric]?.value || 0;
            }

            // For error metrics (MAE, RMSE, KS, Wasserstein), lower is better
            const errorMetrics = ['grade_mae', 'grade_rmse', 'ks_statistic', 'wasserstein_distance'];
            if (errorMetrics.includes(this.currentMetric)) {
                return scoreA - scoreB;
            } else {
                return scoreB - scoreA;
            }
        });
    }

    updateTableHeaders() {
        // Update active header
        document.querySelectorAll('.clickable').forEach(header => {
            header.classList.remove('active');
        });

        const activeHeader = document.querySelector(`[data-dataset="${this.currentSortBy}"]`);
        if (activeHeader) {
            activeHeader.classList.add('active');
        }

        // Update metric display in headers
        const metricDisplay = this.getMetricDisplayName(this.currentMetric);
        document.querySelectorAll('.clickable').forEach(header => {
            const text = header.textContent.trim().split(' ')[0];
            header.innerHTML = `${text} <i class="fas fa-sort"></i>`;
        });
    }

    populateTableBody(sortedData) {
        const tbody = document.getElementById('leaderboardBody');
        tbody.innerHTML = '';

        sortedData.forEach((run, index) => {
            const row = this.createTableRow(run, index + 1);
            tbody.appendChild(row);
        });
    }

    createTableRow(run, rank) {
        const row = document.createElement('tr');
        
        // Rank with medal
        const rankCell = document.createElement('td');
        rankCell.className = 'rank-col';
        rankCell.innerHTML = this.createRankDisplay(rank);
        row.appendChild(rankCell);

        // Parameters (including Run ID)
        const paramsCell = document.createElement('td');
        paramsCell.className = 'params-col';
        paramsCell.innerHTML = this.createParametersDisplay(run.parameters, run.runId);
        row.appendChild(paramsCell);

        // Overall score
        const overallCell = document.createElement('td');
        overallCell.className = 'overall-col';
        overallCell.innerHTML = this.createScoreDisplay(run.overall[this.currentMetric]);
        row.appendChild(overallCell);

        // Dataset scores
        this.datasets.forEach(dataset => {
            const datasetCell = document.createElement('td');
            datasetCell.className = 'dataset-col';
            datasetCell.innerHTML = this.createScoreDisplay(run.datasets[dataset][this.currentMetric]);
            row.appendChild(datasetCell);
        });

        return row;
    }

    createRankDisplay(rank) {
        if (rank === 1) {
            return '<div class="rank-medal gold"><i class="fas fa-medal"></i></div>';
        } else if (rank === 2) {
            return '<div class="rank-medal silver"><i class="fas fa-medal"></i></div>';
        } else if (rank === 3) {
            return '<div class="rank-medal bronze"><i class="fas fa-medal"></i></div>';
        } else {
            return `<div class="rank-number">${rank}</div>`;
        }
    }

    createParametersDisplay(params, runId) {
        const display = document.createElement('div');
        display.className = 'params-display';

        const items = [
            { label: 'Run ID', value: runId },
            { label: 'Model', value: params.model_name },
            { label: 'Guidance', value: params.show_guidance ? 'Yes' : 'No', type: 'badge' },
            { label: 'Explain', value: params.explanation ? 'Yes' : 'No', type: 'badge' },
            { label: 'Few-shot', value: params.training_examples },
            { label: 'Samples', value: params.test_samples }
        ];

        items.forEach(item => {
            const itemDiv = document.createElement('div');
            itemDiv.className = 'param-item';

            const labelSpan = document.createElement('span');
            labelSpan.className = 'param-label';
            labelSpan.textContent = item.label + ':';

            const valueSpan = document.createElement('span');
            if (item.type === 'badge') {
                valueSpan.className = `param-badge ${item.value.toLowerCase() === 'yes'}`;
                valueSpan.textContent = item.value;
            } else {
                valueSpan.className = 'param-value';
                valueSpan.textContent = item.value;
            }

            itemDiv.appendChild(labelSpan);
            itemDiv.appendChild(valueSpan);
            display.appendChild(itemDiv);
        });

        return display.outerHTML;
    }

    createScoreDisplay(metric) {
        if (!metric || metric.value === undefined) {
            return '<div class="score-cell"><span class="score-value">-</span></div>';
        }

        const display = document.createElement('div');
        display.className = 'score-cell';

        const valueSpan = document.createElement('span');
        valueSpan.className = 'score-value';
        valueSpan.textContent = this.formatScore(metric.value);
        
        // Add color coding based on metric type and value
        const colorClass = this.getScoreColorClass(metric.value);
        if (colorClass) {
            valueSpan.classList.add(colorClass);
        }

        const stderrSpan = document.createElement('span');
        stderrSpan.className = 'score-stderr';
        stderrSpan.textContent = `±${this.formatScore(metric.stderr)}`;

        display.appendChild(valueSpan);
        display.appendChild(stderrSpan);

        return display.outerHTML;
    }

    formatScore(value) {
        if (value === undefined || value === null) return '-';
        
        // Format based on metric type
        if (this.currentMetric === 'exact_grade_match') {
            return (value * 100).toFixed(1) + '%';
        } else if (['pearson_correlation', 'spearman_correlation'].includes(this.currentMetric)) {
            return value.toFixed(3);
        } else {
            return value.toFixed(3);
        }
    }

    getScoreColorClass(value) {
        if (value === undefined || value === null) return null;

        // Color coding based on metric type
        if (['pearson_correlation', 'spearman_correlation', 'exact_grade_match'].includes(this.currentMetric)) {
            if (value >= 0.7) return 'high';
            if (value >= 0.4) return 'medium';
            return 'low';
        } else if (['grade_mae', 'grade_rmse', 'ks_statistic', 'wasserstein_distance'].includes(this.currentMetric)) {
            if (value <= 0.3) return 'high';
            if (value <= 0.6) return 'medium';
            return 'low';
        }
        return null;
    }

    getMetricDisplayName(metric) {
        const names = {
            'pearson_correlation': 'Pearson Correlation',
            'spearman_correlation': 'Spearman Correlation',
            'exact_grade_match': 'Exact Grade Match',
            'grade_mae': 'Mean Absolute Error',
            'grade_rmse': 'Root Mean Square Error',
            'ks_statistic': 'Kolmogorov-Smirnov',
            'wasserstein_distance': 'Wasserstein Distance'
        };
        return names[metric] || metric;
    }

    showLoading(show) {
        const loading = document.getElementById('loadingIndicator');
        loading.style.display = show ? 'block' : 'none';
    }

    showError(message) {
        const errorDiv = document.getElementById('errorMessage');
        const errorText = document.getElementById('errorText');
        errorText.textContent = message;
        errorDiv.style.display = 'block';
    }

    hideError() {
        const errorDiv = document.getElementById('errorMessage');
        errorDiv.style.display = 'none';
    }
}

// Initialize the leaderboard when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new MentorEvalLeaderboard();
});
