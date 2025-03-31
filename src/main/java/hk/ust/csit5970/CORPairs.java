package hk.ust.csit5970;

import org.apache.commons.cli.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import java.io.*;
import java.net.URI;
import java.util.*;

/**
 * Compute the bigram count using "pairs" approach.
 * This program uses a two-pass MapReduce job to calculate the correlation between word pairs.
 */
public class CORPairs extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(CORPairs.class);

    /*
     * First-pass Mapper: Tokenizes input text and counts word occurrences.
     */
    private static class CORMapper1 extends Mapper<LongWritable, Text, Text, IntWritable> {
        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            HashMap<String, Integer> wordCount = new HashMap<>();
            // Clean the input text and tokenize it
            String clean_doc = value.toString().replaceAll("[^a-z A-Z]", " ");
            StringTokenizer doc_tokenizer = new StringTokenizer(clean_doc);

            // Count occurrences of each word
            while (doc_tokenizer.hasMoreTokens()) {
                String token = doc_tokenizer.nextToken().toLowerCase();
                wordCount.put(token, wordCount.getOrDefault(token, 0) + 1);
            }

            // Emit each word with its count
            for (Map.Entry<String, Integer> entry : wordCount.entrySet()) {
                context.write(new Text(entry.getKey()), new IntWritable(entry.getValue()));
            }
        }
    }

    /*
     * First-pass Reducer: Aggregates word counts from all mappers.
     */
    private static class CORReducer1 extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            // Sum up counts for each word
            for (IntWritable val : values) {
                sum += val.get();
            }
            context.write(key, new IntWritable(sum));
        }
    }

    /*
     * Second-pass Mapper: Generates unique word pairs (A, B) where A < B.
     */
    public static class CORPairsMapper2 extends Mapper<LongWritable, Text, PairOfStrings, IntWritable> {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // Tokenize the input text
            StringTokenizer doc_tokenizer = new StringTokenizer(value.toString().replaceAll("[^a-z A-Z]", " "));
            Set<String> uniqueWords = new TreeSet<>();

            // Collect unique words
            while (doc_tokenizer.hasMoreTokens()) {
                String token = doc_tokenizer.nextToken().toLowerCase();
                if (!token.isEmpty()) {
                    uniqueWords.add(token);
                }
            }

            // Generate all unique pairs (A, B) with A < B
            List<String> wordsList = new ArrayList<>(uniqueWords);
            int n = wordsList.size();
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    PairOfStrings pair = new PairOfStrings(wordsList.get(i), wordsList.get(j));
                    context.write(pair, new IntWritable(1));
                }
            }
        }
    }

    /*
     * Second-pass Combiner: Combines intermediate results to reduce data transfer.
     */
    private static class CORPairsCombiner2 extends Reducer<PairOfStrings, IntWritable, PairOfStrings, IntWritable> {
        @Override
        protected void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            // Sum up counts for each pair
            for (IntWritable val : values) {
                sum += val.get();
            }
            context.write(key, new IntWritable(sum));
        }
    }

    /*
     * Second-pass Reducer: Calculates correlation for each word pair.
     */
    public static class CORPairsReducer2 extends Reducer<PairOfStrings, IntWritable, PairOfStrings, DoubleWritable> {
        private final static Map<String, Integer> word_total_map = new HashMap<>();

        /*
         * Preload the intermediate result file containing word frequencies.
         */
        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Path middle_result_path = new Path("mid/part-r-00000");
            Configuration middle_conf = new Configuration();
            try {
                FileSystem fs = FileSystem.get(URI.create(middle_result_path.toString()), middle_conf);

                if (!fs.exists(middle_result_path)) {
                    throw new IOException(middle_result_path.toString() + " not exist!");
                }

                // Read the intermediate result file
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(middle_result_path)))) {
                    LOG.info("Reading intermediate results...");
                    String line;
                    while ((line = reader.readLine()) != null) {
                        String[] line_terms = line.split("\t");
                        word_total_map.put(line_terms[0], Integer.valueOf(line_terms[1]));
                    }
                }
                LOG.info("Finished reading intermediate results!");
            } catch (Exception e) {
                System.out.println(e.getMessage());
            }
        }

        /*
         * Reduce function: Calculate correlation for each word pair.
         */
        @Override
        protected void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int pairCount = 0;
            // Sum up counts for the pair
            for (IntWritable val : values) {
                pairCount += val.get();
            }

            String wordA = key.getLeftElement();
            String wordB = key.getRightElement();
            Integer freqA = word_total_map.get(wordA);
            Integer freqB = word_total_map.get(wordB);

            // Skip if frequencies are missing or zero
            if (freqA == null || freqB == null || freqA == 0 || freqB == 0) {
                return;
            }

            // Calculate correlation
            double correlation = (double) pairCount / (freqA * freqB);
            context.write(key, new DoubleWritable(correlation));
        }
    }

    /*
     * Custom Partitioner: Ensures pairs with the same left element go to the same reducer.
     */
    private static final class MyPartitioner extends Partitioner<PairOfStrings, FloatWritable> {
        @Override
        public int getPartition(PairOfStrings key, FloatWritable value, int numReduceTasks) {
            return (key.getLeftElement().hashCode() & Integer.MAX_VALUE) % numReduceTasks;
        }
    }

    /*
     * Main driver method for the MapReduce job.
     */
    @SuppressWarnings({ "static-access" })
    public int run(String[] args) throws Exception {
        Options options = new Options();

        // Define command-line options
        options.addOption(OptionBuilder.withArgName("path").hasArg().withDescription("input path").create(INPUT));
        options.addOption(OptionBuilder.withArgName("path").hasArg().withDescription("output path").create(OUTPUT));
        options.addOption(OptionBuilder.withArgName("num").hasArg().withDescription("number of reducers").create(NUM_REDUCERS));

        CommandLine cmdline;
        CommandLineParser parser = new GnuParser();

        try {
            cmdline = parser.parse(options, args);
        } catch (ParseException exp) {
            System.err.println("Error parsing command line: " + exp.getMessage());
            return -1;
        }

        // Validate arguments
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

        int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

        LOG.info("Tool: " + CORPairs.class.getSimpleName());
        LOG.info(" - input path: " + inputPath);
        LOG.info(" - output path: " + outputPath);
        LOG.info(" - number of reducers: " + reduceTasks);

        // First-pass MapReduce job
        Configuration conf1 = new Configuration();
        Job job1 = Job.getInstance(conf1, "Firstpass");

        job1.setJarByClass(CORPairs.class);
        job1.setMapperClass(CORMapper1.class);
        job1.setReducerClass(CORReducer1.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(IntWritable.class);

        FileInputFormat.setInputPaths(job1, new Path(inputPath));
        FileOutputFormat.setOutputPath(job1, new Path(middlePath));

        // Delete the intermediate output directory if it exists
        Path middleDir = new Path(middlePath);
        FileSystem.get(conf1).delete(middleDir, true);

        long startTime = System.currentTimeMillis();
        job1.waitForCompletion(true);
        LOG.info("Job 1 Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

        // Second-pass MapReduce job
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

        // Delete the final output directory if it exists
        Path outputDir = new Path(outputPath);
        FileSystem.get(conf1).delete(outputDir, true);

        startTime = System.currentTimeMillis();
        job2.waitForCompletion(true);
        LOG.info("Job 2 Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

        return 0;
    }

    /**
     * Main method to run the tool.
     */
    public static void main(String[] args) throws Exception {
        ToolRunner.run(new CORPairs(), args);
    }
}