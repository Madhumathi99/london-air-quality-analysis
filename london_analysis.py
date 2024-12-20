import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class LondonAirQualityAnalysis:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, 'results')
        self.viz_dir = os.path.join(self.results_dir, 'visualizations')
        
        # Create necessary directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Set style for visualizations
        plt.style.use('default')  # Using default style instead of seaborn
        sns.set_theme()  # Set seaborn theme

    def _preprocess_emissions_data(self, df):
        """Preprocess the emissions data"""
        print("Starting data preprocessing...")
        
        # Get all year columns
        year_cols = []
        for year in ['2019', '2025', '2030']:
            year_cols.extend([col for col in df.columns if col.endswith(year)])
        
        # Melt the dataframe
        id_vars = ['TOID', 'pollutant', 'emissions-units']
        melted_df = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=year_cols,
            var_name='vehicle_year',
            value_name='emissions'
        )
        
        # Extract information
        melted_df['year'] = melted_df['vehicle_year'].str.extract(r'(\d{4})').astype(int)
        melted_df['vehicle_type'] = melted_df['vehicle_year'].str.split('-').str[1:-1].str.join('-')
        melted_df['emissions'] = pd.to_numeric(melted_df['emissions'], errors='coerce')
        
        print("Columns after preprocessing:", melted_df.columns.tolist())
        return melted_df

    def load_data(self):
        """Load the emissions data"""
        print("\nLoading data...")
        file_path = os.path.join(self.input_dir, "LAEI2019-nox-pm-co2-major-roads-link-emissions.xlsx")
        
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            print(f"Loaded {len(df)} rows")
            print("Original columns:", df.columns.tolist())
            
            processed_df = self._preprocess_emissions_data(df)
            print(f"Processed data shape: {processed_df.shape}")
            
            return processed_df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def clean_data(self, df):
        """Clean the data"""
        print("\nCleaning data...")
        
        # Remove rows with null emissions
        df_cleaned = df.dropna(subset=['emissions'])
        
        # Remove outliers using percentiles
        lower_bound = df_cleaned['emissions'].quantile(0.01)
        upper_bound = df_cleaned['emissions'].quantile(0.99)
        mask = (df_cleaned['emissions'] >= lower_bound) & (df_cleaned['emissions'] <= upper_bound)
        df_cleaned = df_cleaned[mask]
        
        print(f"Data shape after cleaning: {df_cleaned.shape}")
        return df_cleaned

    def analyze_data(self, df):
        """Perform various analyses"""
        print("\nPerforming analysis...")
        analyses = {}

        # 1. Temporal Analysis
        analyses['temporal'] = df.groupby(['year', 'pollutant'])['emissions'].agg([
            'sum', 'mean', 'std'
        ]).round(2).reset_index()

        # 2. Vehicle Type Analysis
        analyses['vehicle'] = df.groupby(['vehicle_type', 'year'])['emissions'].agg([
            'sum', 'mean', 'count'
        ]).round(2).reset_index()

        # 3. Pollutant Analysis
        analyses['pollutant'] = df.groupby(['pollutant', 'year'])['emissions'].agg([
            'sum', 'mean', 'std'
        ]).round(2).reset_index()

        # 4. Top Contributors
        analyses['top_contributors'] = df.groupby(['vehicle_type', 'pollutant'])['emissions'].sum()\
            .sort_values(ascending=False)\
            .reset_index()

        return analyses

    def create_visualizations(self, df, analyses):
        """Create and save visualizations"""
        print("\nGenerating visualizations...")
        
        try:
            # 1. Temporal Trends Line Plot
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=analyses['temporal'], x='year', y='sum', hue='pollutant', marker='o')
            plt.title('Emissions Trends Over Time')
            plt.xlabel('Year')
            plt.ylabel('Total Emissions')
            plt.xticks(analyses['temporal']['year'].unique())
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, '1_temporal_trends.png'))
            plt.close()

            # 2. Vehicle Type Emissions Bar Plot
            plt.figure(figsize=(15, 7))
            vehicle_pivot = analyses['vehicle'].pivot(index='vehicle_type', columns='year', values='sum')
            vehicle_pivot.plot(kind='bar', width=0.8)
            plt.title('Emissions by Vehicle Type and Year')
            plt.xlabel('Vehicle Type')
            plt.ylabel('Total Emissions')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Year')
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, '2_vehicle_emissions.png'))
            plt.close()

            # 3. Pollutant Distribution Box Plot
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x='pollutant', y='emissions')
            plt.title('Distribution of Emissions by Pollutant Type')
            plt.xlabel('Pollutant')
            plt.ylabel('Emissions')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, '3_pollutant_distribution.png'))
            plt.close()

            # 4. Top Contributors Heat Map
            plt.figure(figsize=(12, 8))
            top_pivot = analyses['top_contributors'].pivot(
                index='vehicle_type', 
                columns='pollutant', 
                values='emissions'
            )
            sns.heatmap(top_pivot, annot=True, fmt='.0f', cmap='YlOrRd')
            plt.title('Emissions Heat Map: Vehicle Types vs Pollutants')
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, '4_emissions_heatmap.png'))
            plt.close()

            # 5. Year-wise Emissions Distribution
            plt.figure(figsize=(12, 6))
            sns.violinplot(data=df, x='year', y='emissions')
            plt.title('Distribution of Emissions Across Years')
            plt.xlabel('Year')
            plt.ylabel('Emissions')
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, '5_yearly_distribution.png'))
            plt.close()

            # 6. Vehicle Type Contribution Pie Chart
            plt.figure(figsize=(10, 8))
            vehicle_total = analyses['vehicle'].groupby('vehicle_type')['sum'].sum()
            plt.pie(vehicle_total, labels=vehicle_total.index, autopct='%1.1f%%')
            plt.title('Vehicle Type Contribution to Total Emissions')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, '6_vehicle_contribution_pie.png'))
            plt.close()

            # 7. Emission Trends by Vehicle Category
            plt.figure(figsize=(15, 8))
            top_vehicles = df.groupby('vehicle_type')['emissions'].sum().nlargest(5).index
            vehicle_trends = df[df['vehicle_type'].isin(top_vehicles)]
            sns.lineplot(data=vehicle_trends, x='year', y='emissions', hue='vehicle_type')
            plt.title('Emission Trends for Top 5 Vehicle Categories')
            plt.xlabel('Year')
            plt.ylabel('Emissions')
            plt.xticks(df['year'].unique())
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, '7_vehicle_trends.png'))
            plt.close()

            # 8. Yearly Pollutant Comparison
            plt.figure(figsize=(12, 6))
            yearly_comparison = df.pivot_table(
                values='emissions', 
                index='year',
                columns='pollutant', 
                aggfunc='mean'
            )
            yearly_comparison.plot(kind='bar', width=0.8)
            plt.title('Average Emissions by Year and Pollutant')
            plt.xlabel('Year')
            plt.ylabel('Average Emissions')
            plt.legend(title='Pollutant', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, '8_yearly_pollutant_comparison.png'))
            plt.close()

            print(f"Saved all visualizations to {self.viz_dir}")
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            raise

    def save_results(self, analyses):
        """Save analysis results"""
        print("\nSaving results...")
        
        for name, df in analyses.items():
            output_file = os.path.join(self.results_dir, f'{name}_analysis.csv')
            df.to_csv(output_file, index=False)
            print(f"Saved {name} analysis to {output_file}")

    def generate_report(self, analyses):
        """Generate a summary report"""
        print("\nGenerating report...")
        
        report = []
        report.append("=== London Air Quality Analysis Report ===")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for name, df in analyses.items():
            report.append(f"\n{name.upper()} ANALYSIS")
            report.append("-" * 50)
            report.append(df.head().to_string())
            report.append("\nKey Statistics:")
            if 'sum' in df.columns:
                report.append(f"Total emissions: {df['sum'].sum():,.2f}")
            report.append("-" * 50)
        
        report.append("\nVISUALIZATIONS GENERATED")
        report.append("-" * 50)
        report.append("1. Temporal Trends: Line plot showing emission trends over time")
        report.append("2. Vehicle Emissions: Bar plot comparing emissions by vehicle type")
        report.append("3. Pollutant Distribution: Box plot showing emission distributions")
        report.append("4. Emissions Heat Map: Heat map of vehicle types vs pollutants")
        report.append("5. Yearly Distribution: Violin plot of emissions distribution by year")
        report.append("6. Vehicle Contribution: Pie chart of vehicle type contributions")
        report.append("7. Vehicle Trends: Line plot showing trends for top vehicle categories")
        report.append("8. Yearly Pollutant Comparison: Bar plot of yearly emissions by pollutant")
        
        report_path = os.path.join(self.results_dir, 'analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        print(f"Report saved to {report_path}")

    def run_pipeline(self):
        """Execute the complete analysis pipeline"""
        print("Starting London Air Quality Analysis Pipeline...")
        
        try:
            # Load Data
            df = self.load_data()
            
            # Clean Data
            df_cleaned = self.clean_data(df)
            
            # Analyze Data
            analyses = self.analyze_data(df_cleaned)
            
            # Create Visualizations
            self.create_visualizations(df_cleaned, analyses)
            
            # Save Results
            self.save_results(analyses)
            
            # Generate Report
            self.generate_report(analyses)
            
            print("\nPipeline completed successfully!")
            return analyses
            
        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            raise

def main():
    # Define directories
    input_dir = "E:/London Quality Analysis/LAEI2019-Concentrations-Data-CSV"
    output_dir = "E:/London Quality Analysis/output"
    
    # Run the pipeline
    pipeline = LondonAirQualityAnalysis(input_dir, output_dir)
    analyses = pipeline.run_pipeline()

if __name__ == "__main__":
    main()