# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy import stats
    from scipy.stats import pearsonr, spearmanr
    return mo, np, pd, plt, sns, stats


@app.cell
def _(mo):
    mo.md("""
    # Pull-Up Training Analysis
    """)
    return


@app.cell
def _(pd):
    # Load the data
    df_raw = pd.read_csv('data/pullup_logs.csv')
    df_raw
    return (df_raw,)


@app.cell
def _(df_raw, pd):
    # Clean the data
    df = df_raw.copy()

    # Clean column names - remove extra whitespace and newlines
    df.columns = df.columns.str.replace('\n', ' ').str.strip()

    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

    # Remove rows where all workout data is missing
    workout_columns = [col for col in df.columns if col != 'Date']
    df_cleaned = df.dropna(how='all', subset=workout_columns)

    # Remove Notes Column
    df_cleaned = df_cleaned.drop(columns=['Unnamed: 13'])

    # Sort by date
    df_cleaned = df_cleaned.sort_values('Date').reset_index(drop=True)

    # Add day number for trend analysis
    df_cleaned['Day_Number'] = range(1, len(df_cleaned) + 1)

    df_cleaned
    return (df_cleaned,)


@app.cell
def _(mo):
    mo.md("""
    ---
    # 📊 Dataset Overview & Descriptive Statistics
    """)
    return


@app.cell
def _(df_cleaned, mo):
    mo.md(f"""
    ## Summary Information
    - **Total workout days:** {len(df_cleaned)}
    - **Date range:** {df_cleaned['Date'].min().strftime('%B %d, %Y')} to {df_cleaned['Date'].max().strftime('%B %d, %Y')}
    - **Training duration:** {(df_cleaned['Date'].max() - df_cleaned['Date'].min()).days} days
    - **First pull-up achieved:** {df_cleaned[df_cleaned['Maximum  Pull-Ups'] > 0]['Date'].min().strftime('%B %d, %Y')}
    """)
    return


@app.cell
def _(df_cleaned):
    df_cleaned.drop(columns=['Date', 'Day_Number']).describe()
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # Research Question 1: Does Daily Dead-Hang Training Improve Pull-Up Performance?

    **H₀:** Daily dead-hang training does NOT result in statistically significant improvement in pull-up performance.

    **H₁:** Daily dead-hang training DOES result in statistically significant improvement in pull-up performance.
    """)
    return


@app.cell
def _(df_cleaned, stats):
    # Statistical Analysis for Q1
    # Correlation between dead hang and pull-ups
    valid_data = df_cleaned[['Average Dead  Hang (secs)', 'Maximum  Pull-Ups']].dropna()

    correlation, p_value_corr = stats.pearsonr(
        valid_data['Average Dead  Hang (secs)'], 
        valid_data['Maximum  Pull-Ups']
    )

    # Linear regression for trend
    from scipy.stats import linregress
    slope, intercept, r_value, p_value_trend, std_err = linregress(
        df_cleaned['Day_Number'], 
        df_cleaned['Average Dead  Hang (secs)']
    )

    # Pull-up progression trend
    pullup_data = df_cleaned[df_cleaned['Maximum  Pull-Ups'] > 0]
    if len(pullup_data) > 1:
        slope_pullup, intercept_pullup, r_value_pullup, p_value_pullup, std_err_pullup = linregress(
            pullup_data['Day_Number'], 
            pullup_data['Maximum  Pull-Ups']
        )
    else:
        slope_pullup, p_value_pullup, r_value_pullup = 0, 1, 0

    q1_results = {
        'correlation': correlation,
        'p_value_corr': p_value_corr,
        'slope': slope,
        'p_value_trend': p_value_trend,
        'r_squared': r_value**2,
        'slope_pullup': slope_pullup,
        'p_value_pullup': p_value_pullup,
        'r_squared_pullup': r_value_pullup**2
    }
    return intercept, intercept_pullup, q1_results, slope, slope_pullup


@app.cell
def _(mo, q1_results):
    mo.md(f"""
    ## Statistical Results - Question 1

    ### Correlation Analysis
    - **Pearson Correlation** between Dead Hang and Pull-Ups: **r = {q1_results['correlation']:.3f}**
    - **P-value:** {q1_results['p_value_corr']:.4f}
    - **Interpretation:** {"Strong positive correlation" if abs(q1_results['correlation']) > 0.7 else "Moderate positive correlation" if abs(q1_results['correlation']) > 0.4 else "Weak correlation"}

    ### Trend Analysis
    - **Dead Hang Improvement Rate:** {q1_results['slope']:.3f} seconds per day
    - **R² (Dead Hang):** {q1_results['r_squared']:.3f} ({q1_results['r_squared']*100:.1f}% variance explained)
    - **P-value (trend):** {q1_results['p_value_trend']:.4f}

    ### Pull-Up Performance
    - **Pull-Up Improvement Rate:** {q1_results['slope_pullup']:.4f} reps per day
    - **R² (Pull-Ups):** {q1_results['r_squared_pullup']:.3f}
    - **P-value (pull-up trend):** {q1_results['p_value_pullup']:.4f}

    ### Conclusion
    {" **REJECT NULL HYPOTHESIS** - Dead-hang training shows statistically significant improvement in pull-up performance (p < 0.05)" if q1_results['p_value_corr'] < 0.05 and q1_results['correlation'] > 0 else "❌ **FAIL TO REJECT NULL HYPOTHESIS** - No statistically significant evidence of improvement"}
    """)
    return


@app.cell
def _(df_cleaned, intercept, intercept_pullup, np, plt, slope, slope_pullup):
    # Visualization for Q1
    def create_q1_visualizations():
        fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Dead Hang Progress Over Time
        ax1 = axes1[0, 0]
        ax1.plot(df_cleaned['Date'], df_cleaned['Average Dead  Hang (secs)'], 
                 marker='o', linewidth=2, markersize=6, color='#2E86AB', label='Actual')
        # Add trend line
        x_vals = df_cleaned['Day_Number']
        y_vals = intercept + slope * x_vals
        ax1.plot(df_cleaned['Date'], y_vals, '--', color='red', linewidth=2, label=f'Trend (slope={slope:.3f})')
        ax1.set_title('Dead Hang Performance Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Average Dead Hang (seconds)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # 2. Pull-Ups Progress Over Time
        ax2 = axes1[0, 1]
        ax2.plot(df_cleaned['Date'], df_cleaned['Maximum  Pull-Ups'], 
                 marker='s', linewidth=2, markersize=6, color='#A23B72', label='Actual')
        # Add trend line for pull-ups
        pullup_mask = df_cleaned['Maximum  Pull-Ups'] > 0
        if pullup_mask.any():
            x_pullup = df_cleaned[pullup_mask]['Day_Number']
            y_pullup_trend = intercept_pullup + slope_pullup * x_pullup
            ax2.plot(df_cleaned[pullup_mask]['Date'], y_pullup_trend, '--', 
                    color='red', linewidth=2, label=f'Trend (slope={slope_pullup:.4f})')
        ax2.set_title('Pull-Up Performance Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Maximum Pull-Ups', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        # 3. Correlation Scatter Plot
        ax3 = axes1[1, 0]
        scatter = ax3.scatter(df_cleaned['Average Dead  Hang (secs)'], 
                             df_cleaned['Maximum  Pull-Ups'],
                             c=df_cleaned['Day_Number'], cmap='viridis', 
                             s=100, alpha=0.6, edgecolors='black')
        # Add trend line
        z = np.polyfit(df_cleaned['Average Dead  Hang (secs)'], df_cleaned['Maximum  Pull-Ups'], 1)
        p = np.poly1d(z)
        ax3.plot(df_cleaned['Average Dead  Hang (secs)'], 
                 p(df_cleaned['Average Dead  Hang (secs)']), 
                 "r--", linewidth=2, label=f'Linear fit')
        ax3.set_title('Dead Hang vs Pull-Ups Correlation', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Average Dead Hang (seconds)', fontsize=12)
        ax3.set_ylabel('Maximum Pull-Ups', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Day Number')

        # 4. Weekly Rolling Average
        ax4 = axes1[1, 1]
        df_cleaned['Dead_Hang_MA7'] = df_cleaned['Average Dead  Hang (secs)'].rolling(window=7, min_periods=1).mean()
        ax4.plot(df_cleaned['Date'], df_cleaned['Average Dead  Hang (secs)'], 
                 alpha=0.3, color='gray', label='Daily')
        ax4.plot(df_cleaned['Date'], df_cleaned['Dead_Hang_MA7'], 
                 linewidth=3, color='#F18F01', label='7-Day Moving Average')
        ax4.set_title('Dead Hang - Daily vs 7-Day Moving Average', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Date', fontsize=12)
        ax4.set_ylabel('Average Dead Hang (seconds)', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig1

    fig1 = create_q1_visualizations()
    fig1
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # Research Question 2: Body Composition & Performance

    **H₀:** Changes in body composition or muscle size are NOT significantly associated with improvements in pull-up performance.

    **H₁:** Changes in body composition or muscle size ARE significantly associated with improvements in pull-up performance.
    """)
    return


@app.cell
def _(df_cleaned, pd, stats):
    # Statistical Analysis for Q2
    # Calculate correlations with pull-up performance
    body_metrics = [
        'Left Forearm  Circumference (cm)',
        'Right Forearm  Circumference (cm)',
        'Left Biceps  Circumference (cm)',
        'Right Biceps Circumference (cm)',
        'Lats Spread  Width (cm)',
        'Weight (kg)'
    ]

    correlations_q2 = {}
    for metric in body_metrics:
        valid = df_cleaned[[metric, 'Maximum  Pull-Ups']].dropna()
        if len(valid) > 2:
            corr, pval = stats.pearsonr(valid[metric], valid['Maximum  Pull-Ups'])
            correlations_q2[metric] = {'correlation': corr, 'p_value': pval}

    # Create correlation dataframe
    corr_df = pd.DataFrame(correlations_q2).T
    corr_df = corr_df.round(4)
    corr_df = corr_df.sort_values('correlation', ascending=False)

    # Calculate changes over time
    first_measurements = df_cleaned.iloc[0]
    last_measurements = df_cleaned.iloc[-1]

    changes = {}
    for metric in body_metrics:
        start_val = first_measurements[metric]
        end_val = last_measurements[metric]
        change = end_val - start_val
        pct_change = (change / start_val * 100) if start_val != 0 else 0
        changes[metric] = {'start': start_val, 'end': end_val, 'change': change, 'pct_change': pct_change}

    changes_df = pd.DataFrame(changes).T
    changes_df = changes_df.round(2)
    return body_metrics, changes_df, corr_df


@app.cell
def _(mo):
    mo.md(f"""
    ## Statistical Results - Question 2

    ### Correlation with Pull-Up Performance
    """)
    return


@app.cell
def _(corr_df):
    corr_df
    return


@app.cell
def _(mo):
    mo.md("""
    ### Body Composition Changes
    """)
    return


@app.cell
def _(changes_df):
    changes_df
    return


@app.cell
def _(corr_df, mo):
    significant_correlations = corr_df[corr_df['p_value'] < 0.05]

    mo.md(f"""
    ### Conclusion

    {"**REJECT NULL HYPOTHESIS**" if len(significant_correlations) > 0 else " **FAIL TO REJECT NULL HYPOTHESIS**"}

    {f"Found {len(significant_correlations)} body metric(s) with statistically significant correlation to pull-up performance (p < 0.05):" if len(significant_correlations) > 0 else "No statistically significant associations found between body composition changes and pull-up performance."}

    {chr(10).join([f"- **{idx}**: r = {row['correlation']:.3f}, p = {row['p_value']:.4f}" for idx, row in significant_correlations.iterrows()]) if len(significant_correlations) > 0 else ""}
    """)
    return (significant_correlations,)


@app.cell
def _(body_metrics, df_cleaned, plt):
    # Visualization for Q2
    def create_q2_visualizations():
        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
        axes2 = axes2.flatten()

        colors = ['#E63946', '#F4A261', '#2A9D8F', '#264653', '#E76F51', '#8338EC']

        for idx, metric in enumerate(body_metrics):
            ax = axes2[idx]
            ax2_twin = ax.twinx()

            # Plot body metric on left axis
            line1 = ax.plot(df_cleaned['Date'], df_cleaned[metric], 
                           marker='o', linewidth=2, markersize=5, 
                           color=colors[idx], label=metric.replace('  ', ' '))
            ax.set_xlabel('Date', fontsize=10)
            ax.set_ylabel(metric.replace('  ', ' '), fontsize=10, color=colors[idx])
            ax.tick_params(axis='y', labelcolor=colors[idx])
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

            # Plot pull-ups on right axis
            line2 = ax2_twin.plot(df_cleaned['Date'], df_cleaned['Maximum  Pull-Ups'], 
                                 marker='s', linewidth=2, markersize=5, 
                                 color='black', alpha=0.5, label='Pull-Ups')
            ax2_twin.set_ylabel('Maximum Pull-Ups', fontsize=10)

            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left', fontsize=8)

            ax.set_title(f'{metric.replace("  ", " ")} vs Pull-Ups', fontsize=11, fontweight='bold')

        plt.tight_layout()
        return fig2

    fig2 = create_q2_visualizations()
    fig2
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    #  Additional Insights
    """)
    return


@app.cell
def _(df_cleaned, np, plt, sns):
    # Additional visualizations
    def create_additional_insights():
        fig3, axes3 = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Heatmap of correlations
        ax_heat = axes3[0, 0]
        correlation_matrix = df_cleaned[[
            'Average Dead  Hang (secs)',
            'Maximum  Pull-Ups',
            'Left Forearm  Circumference (cm)',
            'Right Forearm  Circumference (cm)',
            'Left Biceps  Circumference (cm)',
            'Right Biceps Circumference (cm)',
            'Lats Spread  Width (cm)',
            'Weight (kg)',
            'Perceived  Difficulty (1-10)'
        ]].corr()

        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, ax=ax_heat, cbar_kws={'label': 'Correlation'})
        ax_heat.set_title('Correlation Matrix of All Metrics', fontsize=12, fontweight='bold')
        plt.setp(ax_heat.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(ax_heat.get_yticklabels(), rotation=0, fontsize=8)

        # 2. Perceived Difficulty vs Performance
        ax_diff = axes3[0, 1]
        scatter_diff = ax_diff.scatter(df_cleaned['Average Dead  Hang (secs)'], 
                                       df_cleaned['Perceived  Difficulty (1-10)'],
                                       c=df_cleaned['Maximum  Pull-Ups'], 
                                       s=100, cmap='RdYlGn_r', alpha=0.6, edgecolors='black')
        ax_diff.set_xlabel('Average Dead Hang (seconds)', fontsize=11)
        ax_diff.set_ylabel('Perceived Difficulty (1-10)', fontsize=11)
        ax_diff.set_title('Perceived Difficulty vs Performance', fontsize=12, fontweight='bold')
        ax_diff.grid(True, alpha=0.3)
        plt.colorbar(scatter_diff, ax=ax_diff, label='Pull-Ups')

        # 3. Distribution of Dead Hang Times
        ax_dist = axes3[1, 0]
        ax_dist.hist(df_cleaned['Average Dead  Hang (secs)'], bins=20, 
                    color='#2E86AB', alpha=0.7, edgecolor='black')
        ax_dist.axvline(df_cleaned['Average Dead  Hang (secs)'].mean(), 
                       color='red', linestyle='--', linewidth=2, label=f'Mean: {df_cleaned["Average Dead  Hang (secs)"].mean():.1f}s')
        ax_dist.axvline(df_cleaned['Average Dead  Hang (secs)'].median(), 
                       color='orange', linestyle='--', linewidth=2, label=f'Median: {df_cleaned["Average Dead  Hang (secs)"].median():.1f}s')
        ax_dist.set_xlabel('Average Dead Hang (seconds)', fontsize=11)
        ax_dist.set_ylabel('Frequency', fontsize=11)
        ax_dist.set_title('Distribution of Dead Hang Performance', fontsize=12, fontweight='bold')
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.3, axis='y')

        # 4. Muscle Growth Comparison
        ax_muscle = axes3[1, 1]
        muscle_metrics = ['Left Forearm  Circumference (cm)', 'Right Forearm  Circumference (cm)',
                         'Left Biceps  Circumference (cm)', 'Right Biceps Circumference (cm)']
        start_vals = [df_cleaned[m].iloc[0] for m in muscle_metrics]
        end_vals = [df_cleaned[m].iloc[-1] for m in muscle_metrics]

        x_pos = np.arange(len(muscle_metrics))
        width = 0.35

        bars1 = ax_muscle.bar(x_pos - width/2, start_vals, width, label='Start', color='#A8DADC')
        bars2 = ax_muscle.bar(x_pos + width/2, end_vals, width, label='Current', color='#457B9D')

        ax_muscle.set_xlabel('Muscle Group', fontsize=11)
        ax_muscle.set_ylabel('Circumference (cm)', fontsize=11)
        ax_muscle.set_title('Muscle Size: Start vs Current', fontsize=12, fontweight='bold')
        ax_muscle.set_xticks(x_pos)
        ax_muscle.set_xticklabels(['L Forearm', 'R Forearm', 'L Biceps', 'R Biceps'], fontsize=9)
        ax_muscle.legend()
        ax_muscle.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig3

    fig3 = create_additional_insights()
    fig3
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # Key Findings Summary
    """)
    return


@app.cell
def _(changes_df, df_cleaned, mo, q1_results, significant_correlations):
    dead_hang_improvement = df_cleaned['Average Dead  Hang (secs)'].iloc[-1] - df_cleaned['Average Dead  Hang (secs)'].iloc[0]
    dead_hang_improvement_pct = (dead_hang_improvement / df_cleaned['Average Dead  Hang (secs)'].iloc[0]) * 100

    pullup_start = df_cleaned['Maximum  Pull-Ups'].iloc[0]
    pullup_current = df_cleaned['Maximum  Pull-Ups'].iloc[-1]
    pullup_improvement = pullup_current - pullup_start

    mo.md(f"""
    ## Performance Improvements

    ### Dead Hang
    - **Initial:** {df_cleaned['Average Dead  Hang (secs)'].iloc[0]:.1f} seconds
    - **Current:** {df_cleaned['Average Dead  Hang (secs)'].iloc[-1]:.1f} seconds
    - **Improvement:** +{dead_hang_improvement:.1f} seconds ({dead_hang_improvement_pct:.1f}%)
    - **Average improvement rate:** {q1_results['slope']:.3f} seconds per day

    ### Pull-Ups
    - **Initial:** {pullup_start:.0f} reps
    - **Current:** {pullup_current:.0f} reps
    - **Improvement:** +{pullup_improvement:.0f} reps
    - **First pull-up achieved on day:** {df_cleaned[df_cleaned['Maximum  Pull-Ups'] > 0].index[0] + 1}

    ### Statistical Significance
    - Dead hang and pull-ups correlation: **r = {q1_results['correlation']:.3f}** (p = {q1_results['p_value_corr']:.4f})
    - {"Statistically significant relationship found" if q1_results['p_value_corr'] < 0.05 else " No statistically significant relationship"}

    ## Body Composition Insights

    ### Most Significant Changes
    {chr(10).join([f"- **{idx}**: {row['change']:.2f} cm ({row['pct_change']:.1f}%)" for idx, row in changes_df.head(3).iterrows()])}

    ### Associations with Performance
    {chr(10).join([f"- **{idx}**: r = {row['correlation']:.3f} (p = {row['p_value']:.4f})" for idx, row in significant_correlations.iterrows()]) if len(significant_correlations) > 0 else "- No significant correlations found"}

    ## Training Insights
    - **Average perceived difficulty:** {df_cleaned['Perceived  Difficulty (1-10)'].mean():.1f}/10
    - **Difficulty trend:** {"Decreasing (getting easier)" if df_cleaned['Perceived  Difficulty (1-10)'].iloc[-5:].mean() < df_cleaned['Perceived  Difficulty (1-10)'].iloc[:5].mean() else "Increasing or stable"}
    - **Weight change:** {changes_df.loc['Weight (kg)', 'change']:.1f} kg ({changes_df.loc['Weight (kg)', 'pct_change']:.1f}%)
    """)
    return


@app.cell
def _(df_cleaned):
    # Save cleaned data
    df_cleaned.to_csv('data/pullup_logs_cleaned.csv', index=False)
    return


if __name__ == "__main__":
    app.run()
