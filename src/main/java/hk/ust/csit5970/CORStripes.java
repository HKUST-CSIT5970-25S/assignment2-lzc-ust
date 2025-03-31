package hk.ust.csit5970;

import org.apache.commons.cli.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;

/**
 * Compute the bigram count using the "stripes" approach.
 * This program uses a two-pass MapReduce job to calculate the correlation between word pairs.
 */
public class CORStripes extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(CORStripes.class);

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
                if (token.isEmpty()) continue;
                wordCount.put(token, wordCount.getOrDefault(token, 0) + 1);
            }

            // Emit each word and its count
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
     * Second-pass Mapper: Generates stripes for each word, where each stripe contains co-occurring words and their counts.
     */
    public static class CORStripesMapper2 extends Mapper<LongWritable, Text, Text, MapWritable> {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            Set<String> sorted_word_set = new TreeSet<>();
            // Clean the input text and tokenize it
            String doc_clean = value.toString().replaceAll("[^a-z A-Z]", " ");
            StringTokenizer doc_tokenizers = new StringTokenizer(doc_clean);

            // Collect unique words
            while (doc_tokenizers.hasMoreTokens()) {
                sorted_word_set.add(doc_tokenizers.nextToken().toLowerCase());
            }

            // Generate stripes for each word
            List<String> wordsList = new ArrayList<>(sorted_word_set);
            int n = wordsList.size();
            for (int i = 0; i < n; i++) {
                String wordA = wordsList.get(i);
                MapWritable stripe = new MapWritable();
                for (int j = i + 1; j < n; j++) {
                    String wordB = wordsList.get(j);
                    Text bText = new Text(wordB);
                    // Increment count for wordB in the stripe
                    if (stripe.containsKey(bText)) {
                        IntWritable count = (IntWritable) stripe.get(bText);
                        count.set(count.get() + 1);
                    } else {
                        stripe.put(bText, new IntWritable(1));
                    }
                }
                if (!stripe.isEmpty()) {
                    context.write(new Text(wordA), stripe);
                }
            }
        }
    }

    /*
     * Second-pass Combiner: Combines stripes for the same word to reduce data transfer.
     */
    public static class CORStripesCombiner2 extends Reducer<Text, MapWritable, Text, MapWritable> {
        @Override
        protected void reduce(Text key, Iterable<MapWritable> values, Context context) throws IOException, InterruptedException {
            MapWritable combinedStripe = new MapWritable();
            // Combine all stripes for the same word
            for (MapWritable stripe : values) {
                for (Writable entryKey : stripe.keySet()) {
                    IntWritable count = (IntWritable) stripe.get(entryKey);
                    if (combinedStripe.containsKey(entryKey)) {
                        IntWritable existing = (IntWritable) combinedStripe.get(entryKey);
                        existing.set(existing.get() + count.get());
                    } else {
                        combinedStripe.put(entryKey, new IntWritable(count.get()));
                    }
                }
            }
            context.write(key, combinedStripe);
        }
    }

    /*
     * Second-pass Reducer: Calculates correlation for each word pair.
     */
    public static class CORStripesReducer2 extends Reducer<Text, MapWritable, PairOfStrings, DoubleWritable> {
        private static final Map<String, Integer> word_total_map = new HashMap<>();

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
        protected void reduce(Text key, Iterable<MapWritable> values, Context context) throws IOException, InterruptedException {
            // Aggregate all stripes for the key (word A)
            MapWritable combinedStripe = new MapWritable();
            for (MapWritable stripe : values) {
                for (Writable mapKey : stripe.keySet()) {
                    IntWritable count = (IntWritable) stripe.get(mapKey);
                    if (combinedStripe.containsKey(mapKey)) {
                        IntWritable existing = (IntWritable) combinedStripe.get(mapKey);
                        existing.set(existing.get() + count.get());
                    } else {
                        combinedStripe.put(mapKey, new IntWritable(count.get()));
                    }
                }
            }

            // For each word B in the aggregated stripe, compute the correlation coefficient
            Integer freqA = word_total_map.get(key.toString());
            if (freqA == null) {
                return;
            }

            for (Writable mapKey : combinedStripe.keySet()) {
                String wordB = mapKey.toString();
                if (key.toString().compareTo(wordB) < 0) { // Ensure A < B
                    Integer freqB = word_total_map.get(wordB);
                    if (freqB != null && freqB > 0) {
                        int freqAB = ((IntWritable) combinedStripe.get(mapKey)).get();
                        double correlation = (double) freqAB / (freqA * freqB);
                        context.write(new PairOfStrings(key.toString(), wordB), new DoubleWritable(correlation));
                    }
                }
            }
        }
    }

    /**
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

        LOG.info("Tool: " + CORStripes.class.getSimpleName());
        LOG.info(" - input path: " + inputPath);
        LOG.info(" - middle path: " + middlePath);
        LOG.info(" - output path: " + outputPath);
        LOG.info(" - number of reducers: " + reduceTasks);

        // First-pass MapReduce job
        Configuration conf1 = new Configuration();
        Job job1 = Job.getInstance(conf1, "Firstpass");

        job1.setJarByClass(CORStripes.class);
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

        job2.setJarByClass(CORStripes.class);
        job2.setMapperClass(CORStripesMapper2.class);
        job2.setCombinerClass(CORStripesCombiner2.class);
        job2.setReducerClass(CORStripesReducer2.class);

        job2.setOutputKeyClass(PairOfStrings.class);
        job2.setOutputValueClass(DoubleWritable.class);
        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(MapWritable.class);
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
        ToolRunner.run(new CORStripes(), args);
    }
}