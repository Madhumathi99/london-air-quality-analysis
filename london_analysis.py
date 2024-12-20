from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

class LondonAirQualityAnalysis:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, 'results')
        self.viz_dir = os.path.join(self.results_dir, 'visualizations')
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Initialize Spark
        self.spark = SparkSession.builder \
            .appName("London Air Quality Analysis") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "100") \
            .getOrCreate()
        
        # Set visualization style
        plt.style.use('default')
        sns.set_theme()

    def _define_schema(self):
        """Define schema for the emissions data"""
        schema = StructType([
            StructField("TOID", StringType(), True),
            StructField("pollutant", StringType(), True),
            StructField("emissions_units", StringType(), True)
        ])
        
        # Add vehicle columns for each year
        vehicle_types = ['Car-Diesel', 'Car-Electric', 'Car-Petrol', 'LGV-Diesel', 
                        'LGV-Electric', 'HGV-Rigid', 'HGV-Articulated', 'TfL-Bus',
                        'Non-TfL-Bus-or-Coach', 'Taxi', 'Motorcycle']
        years = ['2019', '2025', '2030']
        
        for vehicle in vehicle_types:
            for year in years:
                field_name = f"Road_{vehicle}_{year}"
                schema.add(field_name, DoubleType(), True)
                
        return schema

    def load_data(self):
        """Load the emissions data"""
        print("\nLoading data...")
        file_path = os.path.join(self.input_dir, "LAEI2019-nox-pm-co2-major-roads-link-emissions.xlsx")
        
        try:
            # Read Excel file
            df = self.spark.read.format("com.crealytics.spark.excel") \
                .option("header", "true") \
                .option("inferSchema", "false") \
                .schema(self._define_schema()) \
                .load(file_path)
            
            # Transform data structure
            vehicle_cols = [c for c in df.columns if c.startswith("Road_")]
            stack_expr = []
            for col in vehicle_cols:
                parts = col.split("_")
                year = parts[-1]
                vehicle_type = "_".join(parts[1:-1])
                stack_expr.extend([lit(year), lit(vehicle_type), col])
            
            n_cols = len(vehicle_cols)
            unpivot_expr = f"stack({n_cols}, {', '.join(['?'] * (n_cols * 3))}) as (year, vehicle_type, emissions)"
            
            df_transformed = df.selectExpr(
                "TOID", 
                "pollutant", 
                "emissions_units",
                unpivot_expr
            )
            
            # Convert data types
            df_final = df_transformed \
                .withColumn("year", col("year").cast(IntegerType())) \
                .withColumn("emissions", col("emissions").cast(DoubleType()))
            
            print(f"Loaded and transformed {df_final.count()} records")
            return df_final
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def clean_data(self, df):
        """Clean the data using Spark operations"""
        print("\nCleaning data...")
        
        # Remove null emissions
        df_cleaned = df.na.drop(subset=["emissions"])
        
        # Calculate percentiles for outlier removal
        quantiles = df_cleaned.approxQuantile("emissions", [0.01, 0.99], 0.01)
        
        # Filter outliers
        df_cleaned = df_cleaned.filter(
            (col("emissions") >= quantiles[0]) & 
            (col("emissions") <= quantiles[1])
        )
        
        print(f"Records after cleaning: {df_cleaned.count()}")
        return df_cleaned

    def analyze_data(self, df):
        """Perform comprehensive analysis using Spark SQL"""
        print("\nPerforming analysis...")
        analyses = {}
        
        # Register temp view for SQL operations
        df.createOrReplaceTempView("emissions_data")

        # 1. Temporal Analysis
        temporal_sql = """
            SELECT 
                year,
                pollutant,
                ROUND(SUM(emissions), 2) as sum,
                ROUND(AVG(emissions), 2) as mean,
                ROUND(STDDEV(emissions), 2) as std
            FROM emissions_data
            GROUP BY year, pollutant
            ORDER BY year, pollutant
        """
        analyses['temporal'] = self.spark.sql(temporal_sql)

        # 2. Vehicle Analysis
        vehicle_sql = """
            SELECT 
                vehicle_type,
                year,
                ROUND(SUM(emissions), 2) as sum,
                ROUND(AVG(emissions), 2) as mean,
                COUNT(*) as count
            FROM emissions_data
            GROUP BY vehicle_type, year
            ORDER BY vehicle_type, year
        """
        analyses['vehicle'] = self.spark.sql(vehicle_sql)

        # 3. Pollutant Analysis
        pollutant_sql = """
            SELECT 
                pollutant,
                year,
                ROUND(SUM(emissions), 2) as sum,
                ROUND(AVG(emissions), 2) as mean,
                ROUND(STDDEV(emissions), 2) as std
            FROM emissions_data
            GROUP BY pollutant, year
            ORDER BY pollutant, year
        """
        analyses['pollutant'] = self.spark.sql(pollutant_sql)

        # 4. Top Contributors
        contributors_sql = """
            SELECT 
                vehicle_type,
                pollutant,
                ROUND(SUM(emissions), 2) as emissions
            FROM emissions_data
            GROUP BY vehicle_type, pollutant
            ORDER BY emissions DESC
        """
        analyses['top_contributors'] = self.spark.sql(contributors_sql)

        return analyses

    def create_visualizations(self, df, analyses):
        """Create visualizations using Spark DataFrames"""
        print("\nGenerating visualizations...")
        
        try:
            # Convert Spark DataFrames to Pandas for visualization
            temporal_pd = analyses['temporal'].toPandas()
            vehicle_pd = analyses['vehicle'].toPandas()
            pollutant_pd = analyses['pollutant'].toPandas()
            contributors_pd = analyses['top_contributors'].toPandas()
            
            # 1. Temporal Trends
            plt.figure(figsize=(12, 6))
            for pollutant in temporal_pd['pollutant'].unique():
                data = temporal_pd[temporal_pd['pollutant'] == pollutant]
                plt.plot(data['year'], data['sum'], marker='o', label=pollutant)
            plt.title('Emissions Trends Over Time')
            plt.xlabel('Year')
            plt.ylabel('Total Emissions')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, '1_temporal_trends.png'))
            plt.close()

            # 2. Vehicle Emissions by Year
            plt.figure(figsize=(15, 7))
            pivot_data = vehicle_pd.pivot(index='vehicle_type', columns='year', values='sum')
            pivot_data.plot(kind='bar')
            plt.title('Emissions by Vehicle Type and Year')
            plt.xlabel('Vehicle Type')
            plt.ylabel('Total Emissions')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Year')
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, '2_vehicle_emissions.png'))
            plt.close()

            # 3. Pollutant Distribution
            plt.figure(figsize=(10, 6))
            df_pd = df.select('pollutant', 'emissions').toPandas()
            sns.boxplot(data=df_pd, x='pollutant', y='emissions')
            plt.title('Distribution of Emissions by Pollutant Type')
            plt.xlabel('Pollutant')
            plt.ylabel('Emissions')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, '3_pollutant_distribution.png'))
            plt.close()

            # 4. Heat Map
            plt.figure(figsize=(12, 8))
            pivot_table = contributors_pd.pivot(index='vehicle_type', columns='pollutant', values='emissions')
            sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd')
            plt.title('Emissions Heat Map: Vehicle Types vs Pollutants')
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, '4_emissions_heatmap.png'))
            plt.close()

            # 5. Yearly Distribution
            plt.figure(figsize=(12, 6))
            df_pd = df.select('year', 'emissions').toPandas()
            sns.violinplot(data=df_pd, x='year', y='emissions')
            plt.title('Distribution of Emissions Across Years')
            plt.xlabel('Year')
            plt.ylabel('Emissions')
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, '5_yearly_distribution.png'))
            plt.close()

            # 6. Vehicle Type Contribution
            plt.figure(figsize=(10, 8))
            total_by_vehicle = vehicle_pd.groupby('vehicle_type')['sum'].sum()
            plt.pie(total_by_vehicle, labels=total_by_vehicle.index, autopct='%1.1f%%')
            plt.title('Vehicle Type Contribution to Total Emissions')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, '6_vehicle_contribution_pie.png'))
            plt.close()

            # 7. Top 5 Vehicle Emission Trends
            plt.figure(figsize=(15, 8))
            top_5_vehicles = vehicle_pd.groupby('vehicle_type')['sum'].sum().nlargest(5).index
            for vehicle in top_5_vehicles:
                data = vehicle_pd[vehicle_pd['vehicle_type'] == vehicle]
                plt.plot(data['year'], data['sum'], marker='o', label=vehicle)
            plt.title('Emission Trends for Top 5 Vehicle Categories')
            plt.xlabel('Year')
            plt.ylabel('Emissions')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, '7_vehicle_trends.png'))
            plt.close()

            # 8. Yearly Pollutant Comparison
            plt.figure(figsize=(12, 6))
            pivot_data = pollutant_pd.pivot(index='year', columns='pollutant', values='mean')
            pivot_data.plot(kind='bar', width=0.8)
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

    def generate_report(self, analyses):
        """Generate analysis report using Spark DataFrame results"""
        print("\nGenerating report...")
        
        report = []
        report.append("=== London Air Quality Analysis Report ===")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Convert Spark DataFrames to Pandas for reporting
        for name, spark_df in analyses.items():
            pd_df = spark_df.toPandas()
            
            report.append(f"\n{name.upper()} ANALYSIS")
            report.append("-" * 50)
            report.append(pd_df.head().to_string())
            report.append("\nKey Statistics:")
            
            if 'sum' in pd_df.columns:
                total = pd_df['sum'].sum()
                report.append(f"Total emissions: {total:,.2f}")
            
            report.append("-" * 50)
        
        # Add visualization descriptions
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

    def save_results(self, analyses):
        """Save analysis results"""
        print("\nSaving results...")
        
        for name, spark_df in analyses.items():
            # Save as CSV
            output_file = os.path.join(self.results_dir, f'{name}_analysis.csv')
            spark_df.toPandas().to_csv(output_file, index=False)
            
            # Save as Parquet
            parquet_file = os.path.join(self.results_dir, f'{name}_analysis.parquet')
            spark_df.write.mode('overwrite').parquet(parquet_file)
            
            print(f"Saved {name} analysis to CSV and Parquet formats")

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
        finally:
            # Stop Spark session
            self.spark.stop()

def main():
    # Define directories
    input_dir = "E:/London Quality Analysis/LAEI2019-Concentrations-Data-CSV"
    output_dir = "E:/London Quality Analysis/output"
    
    # Run the pipeline
    pipeline = LondonAirQualityAnalysis(input_dir, output_dir)
    analyses = pipeline.run_pipeline()

if __name__ == "__main__":
    main()