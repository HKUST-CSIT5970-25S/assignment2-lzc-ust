package hk.ust.csit5970;

import org.apache.commons.cli.*;
import org.apache.commons.cli.Options;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;


import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;
import java.util.Arrays;
import java.util.HashMap;

/**
 * Compute the bigram count using "pairs" approach
 */
public class CORPairs extends Configured implements Tool {
	private static final Logger LOG = Logger.getLogger(CORPairs.class);

	/*
	 * First-pass Mapper: 输出单词及其计数
	 */
	private static class CORMapper1 extends
			Mapper<LongWritable, Text, Text, IntWritable> {
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			HashMap<String, Integer> word_set = new HashMap<>();
			// 使用提供的 tokenizer
			String clean_doc = value.toString().replaceAll("[^a-z A-Z]", " ");
			StringTokenizer doc_tokenizer = new StringTokenizer(clean_doc);

			while (doc_tokenizer.hasMoreTokens()) {
				String word = doc_tokenizer.nextToken().toLowerCase();
				word_set.put(word, word_set.getOrDefault(word, 0) + 1);
			}

			for (Map.Entry<String, Integer> entry : word_set.entrySet()) {
				context.write(new Text(entry.getKey()), new IntWritable(entry.getValue()));
			}
		}
	}

	/*
	 * First-pass Reducer: 聚合单词计数
	 */
	private static class CORReducer1 extends
			Reducer<Text, IntWritable, Text, IntWritable> {
		@Override
		public void reduce(Text key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable value : values) {
				sum += value.get();
			}
			context.write(key, new IntWritable(sum));
		}
	}


	/*
	 * Second-pass Mapper: 输出单词对及其计数
	 */
	public static class CORPairsMapper2 extends
			Mapper<LongWritable, Text, PairOfStrings, IntWritable> {
		@Override
		protected void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			// 使用提供的 tokenizer
			StringTokenizer doc_tokenizer = new StringTokenizer(value.toString().replaceAll("[^a-z A-Z]", " "));
			List<String> words = new ArrayList<>();

			while (doc_tokenizer.hasMoreTokens()) {
				words.add(doc_tokenizer.nextToken().toLowerCase());
			}

			for (int i = 0; i < words.size(); i++) {
				for (int j = i + 1; j < words.size(); j++) {
					String word1 = words.get(i);
					String word2 = words.get(j);
					if (!word1.equals(word2)) {
						PairOfStrings pair = new PairOfStrings(
								word1.compareTo(word2) < 0 ? word1 : word2,
								word1.compareTo(word2) < 0 ? word2 : word1
						);
						context.write(pair, new IntWritable(1));
					}
				}
			}
		}
	}

	/*
	 * Second-pass Combiner: 聚合单词对计数
	 */
	private static class CORPairsCombiner2 extends
			Reducer<PairOfStrings, IntWritable, PairOfStrings, IntWritable> {
		@Override
		protected void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable value : values) {
				sum += value.get();
			}
			context.write(key, new IntWritable(sum));
		}
	}

	/*
	 * Second-pass Reducer: 计算相关系数
	 */
	public static class CORPairsReducer2 extends
			Reducer<PairOfStrings, IntWritable, PairOfStrings, DoubleWritable> {
		private final static Map<String, Integer> word_total_map = new HashMap<>();

		@Override
		protected void setup(Context context) throws IOException, InterruptedException {
			Path middle_result_path = new Path("mid/part-r-00000");
			Configuration middle_conf = new Configuration();
			FileSystem fs = FileSystem.get(URI.create(middle_result_path.toString()), middle_conf);

			if (!fs.exists(middle_result_path)) {
				throw new IOException(middle_result_path.toString() + " not exist!");
			}

			try (BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(middle_result_path)))) {
				String line;
				while ((line = reader.readLine()) != null) {
					String[] parts = line.split("\t");
					word_total_map.put(parts[0], Integer.parseInt(parts[1]));
				}
			}
		}

		@Override
		protected void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			int freqAB = 0;
			for (IntWritable value : values) {
				freqAB += value.get();
			}

			int freqA = word_total_map.getOrDefault(key.getLeftElement(), 0);
			int freqB = word_total_map.getOrDefault(key.getRightElement(), 0);

			if (freqA > 0 && freqB > 0) {
				double correlation = (double) freqAB / (freqA * freqB);
				context.write(key, new DoubleWritable(correlation));
			}
		}
	}

	private static final class MyPartitioner extends Partitioner<PairOfStrings, FloatWritable> {
		@Override
		public int getPartition(PairOfStrings key, FloatWritable value, int numReduceTasks) {
			return (key.getLeftElement().hashCode() & Integer.MAX_VALUE) % numReduceTasks;
		}
	}

	/**
	 * Creates an instance of this tool.
	 */
	public CORPairs() {
	}

	private static final String INPUT = "input";
	private static final String MIDDLE = "middle";
	private static final String OUTPUT = "output";
	private static final String NUM_REDUCERS = "numReducers";

	/**
	 * Runs this tool.
	 */
	@SuppressWarnings({ "static-access" })
	public int run(String[] args) throws Exception {
		Options options = new Options();

		options.addOption(OptionBuilder.withArgName("path").hasArg()
				.withDescription("input path").create(INPUT));
		options.addOption(OptionBuilder.withArgName("path").hasArg()
				.withDescription("output path").create(OUTPUT));
		options.addOption(OptionBuilder.withArgName("num").hasArg()
				.withDescription("number of reducers").create(NUM_REDUCERS));

		CommandLine cmdline;
		CommandLineParser parser = new GnuParser();

		try {
			cmdline = parser.parse(options, args);
		} catch (ParseException exp) {
			System.err.println("Error parsing command line: "
					+ exp.getMessage());
			return -1;
		}

		// Lack of arguments
		if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)) {
			System.out.println("args: " + Arrays.toString(args));
			HelpFormatter formatter = new HelpFormatter();
			formatter.setWidth(120);
			formatter.printHelp(this.getClass().getName(), options);
			ToolRunner.printGenericCommandUsage(System.out);
			return -1;
		}

		String inputPath = cmdline.getOptionValue(INPUT);
		String middlePath = "mid";
		String outputPath = cmdline.getOptionValue(OUTPUT);

		int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer
				.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

		LOG.info("Tool: " + CORPairs.class.getSimpleName());
		LOG.info(" - input path: " + inputPath);
		LOG.info(" - output path: " + outputPath);
		LOG.info(" - number of reducers: " + reduceTasks);

		// Setup for the first-pass MapReduce
		Configuration conf1 = new Configuration();

		Job job1 = Job.getInstance(conf1, "Firstpass");

		job1.setJarByClass(CORPairs.class);
		job1.setMapperClass(CORMapper1.class);
		job1.setReducerClass(CORReducer1.class);
		job1.setOutputKeyClass(Text.class);
		job1.setOutputValueClass(IntWritable.class);

		FileInputFormat.setInputPaths(job1, new Path(inputPath));
		FileOutputFormat.setOutputPath(job1, new Path(middlePath));

		// Delete the output directory if it exists already.
		Path middleDir = new Path(middlePath);
		FileSystem.get(conf1).delete(middleDir, true);

		// Time the program
		long startTime = System.currentTimeMillis();
		job1.waitForCompletion(true);
		LOG.info("Job 1 Finished in " + (System.currentTimeMillis() - startTime)
				/ 1000.0 + " seconds");

		// Setup for the second-pass MapReduce

		// Delete the output directory if it exists already.
		Path outputDir = new Path(outputPath);
		FileSystem.get(conf1).delete(outputDir, true);


		Configuration conf2 = new Configuration();
		Job job2 = Job.getInstance(conf2, "Secondpass");

		job2.setJarByClass(CORPairs.class);
		job2.setMapperClass(CORPairsMapper2.class);
		job2.setCombinerClass(CORPairsCombiner2.class);
		job2.setReducerClass(CORPairsReducer2.class);

		job2.setOutputKeyClass(PairOfStrings.class);
		job2.setOutputValueClass(DoubleWritable.class);
		job2.setMapOutputValueClass(IntWritable.class);
		job2.setNumReduceTasks(reduceTasks);

		FileInputFormat.setInputPaths(job2, new Path(inputPath));
		FileOutputFormat.setOutputPath(job2, new Path(outputPath));

		// Time the program
		startTime = System.currentTimeMillis();
		job2.waitForCompletion(true);
		LOG.info("Job 2 Finished in " + (System.currentTimeMillis() - startTime)
				/ 1000.0 + " seconds");

		return 0;
	}

	/**
	 * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
	 */
	public static void main(String[] args) throws Exception {
		ToolRunner.run(new CORPairs(), args);
	}
}