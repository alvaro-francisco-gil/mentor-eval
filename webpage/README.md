# MentorEval Leaderboard Webpage

A static, interactive leaderboard for visualizing MentorEval benchmark results, designed for easy GitHub Pages deployment.

## Features

- **Interactive Leaderboard**: Sort and filter results by different metrics and datasets
- **Multiple Metrics**: Support for all MentorEval metrics (Pearson correlation, MAE, RMSE, etc.)
- **Dataset-Specific Analysis**: Click on dataset columns to reorder by specific dataset performance
- **Parameter Display**: View run parameters including guidance, explanations, few-shot examples
- **Responsive Design**: Works on desktop and mobile devices
- **GitHub Pages Ready**: Pure static files, no server required

## Quick Start

### GitHub Pages Deployment (Recommended)

#### Option 1: Automatic Deployment (GitHub Actions)
1. **Push your code** to GitHub - the workflow will automatically deploy
2. **Enable GitHub Pages** in repository settings → Pages → Source: GitHub Actions
3. **Your leaderboard will be live at:**
   ```
   https://yourusername.github.io/mentor-eval/
   ```

#### Option 2: Manual Deployment
1. **Run the deployment script:**
   ```bash
   python scripts/deploy_leaderboard_web.py
   ```
2. **Commit and push:**
   ```bash
   git add .
   git commit -m "Deploy leaderboard"
   git push
   ```
3. **Enable GitHub Pages** in repository settings → Pages → Source: Deploy from a branch → main

### Local Testing

1. Open `index.html` directly in your browser
2. Note: Requires the results and runs folders to be accessible

## Usage

### Changing Metrics
- Use the "Evaluation Metric" dropdown to switch between different metrics
- Default metric is Pearson Correlation
- Available metrics:
  - Pearson Correlation
  - Spearman Correlation
  - Exact Grade Match
  - Mean Absolute Error (MAE)
  - Root Mean Square Error (RMSE)
  - Kolmogorov-Smirnov Statistic
  - Wasserstein Distance

### Sorting Results
- Click on dataset column headers to sort by that specific dataset
- Click on "Overall" header to sort by the full benchmark
- The leaderboard automatically reorders based on the selected metric

### View Options
- **Automatic Loading**: Data loads automatically when the page opens

### Understanding the Display

#### Parameter Columns
- **Run ID**: The experiment run identifier
- **Model**: The language model used (e.g., gpt-4o-mini)
- **Guidance**: Whether grading guidance/rubrics were included
- **Explain**: Whether explanations were requested
- **Few-shot**: Number of few-shot examples used
- **Samples**: Number of test samples evaluated

#### Score Display
- **Main Value**: The metric value (formatted appropriately)
- **±Value**: Standard error
- **Color Coding**:
  - 🟢 Green: High performance
  - 🟡 Yellow: Medium performance
  - 🔴 Red: Low performance

#### Dataset Information
The bottom section provides information about each dataset:
- **Language**: English, Portuguese, or Arabic
- **ISCED Level**: Educational level (1-7)
- **Type**: Essay Writing or Short Answer
- **Samples**: Number of student responses

## File Structure

```
webpage/
├── index.html          # Main webpage
├── style.css           # Styling and layout
├── script.js           # Interactive functionality
├── README.md           # This file
└── assets/             # Images and resources
    ├── mentoreval_logo.png
    ├── mentoreval_logo_nobg.png
    └── mentoreval_architecture.png
```

## Data Loading

The static version loads data directly from your repository:

- **Result Files**: Loads from `../results/` directory
- **Run Parameters**: Loads from `../runs/` directory  
- **No Server Required**: Everything works with GitHub's static file serving

## Customization

### Adding New Metrics
1. Add the metric to the `metricSelect` dropdown in `index.html`
2. Update the `getMetricDisplayName()` function in `script.js`
3. Add color coding logic in `getScoreColorClass()` if needed

### Modifying Dataset Information
Update the dataset cards in `index.html` to reflect changes in your datasets.

### Styling Changes
Modify `style.css` to customize the appearance:
- Colors and gradients
- Layout and spacing
- Responsive breakpoints
- Animation effects

## Building the Webpage

The webpage automatically discovers result files and run configurations. Run the build script whenever you add new results:

### Automatic Build
```bash
cd webpage
python build.py
```

### Windows Build
```cmd
cd webpage
build.bat
```

The build script will:
- ✅ Discover all result files in `../results/`
- ✅ Discover all run configurations in `../runs/`
- ✅ Update `script.js` with the correct file lists
- ✅ Generate fallback parameters for all runs

## Troubleshooting

### Data Not Loading
1. **Run the build script first:** `python build.py`
2. Ensure the `results/` directory contains JSON result files
3. Check that the files are committed to your repository
4. Verify file paths in the JavaScript code
5. Check browser console for loading errors

### CORS Issues
If opening `index.html` directly:
1. Use GitHub Pages deployment instead
2. Or serve via a local web server (e.g., `python -m http.server`)

### Performance Issues
- Large result files may take time to load
- Consider implementing pagination for many results
- Use browser developer tools to monitor loading times

## Browser Compatibility

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Contributing

To add new features or fix issues:
1. Modify the relevant files
2. Test with your result data
3. Ensure responsive design works
4. Update this README if needed

## License

Same as the main MentorEval project (GPL).
