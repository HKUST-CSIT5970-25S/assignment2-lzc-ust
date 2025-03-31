package hk.ust.csit5970;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

/**
 * Compute the bigram count using the "stripes" approach.
 */
public class BigramFrequencyStripes extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(BigramFrequencyStripes.class);

    /*
     * Mapper: Emits <word, stripe> where stripe is a hash map of co-occurring words and their counts.
     */
    private static class MyMapper extends Mapper<LongWritable, Text, Text, HashMapStringIntWritable> {

        // Reuse objects to save overhead of object creation.
        private static final Text KEY = new Text();
        private static final HashMapStringIntWritable STRIPE = new HashMapStringIntWritable();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // Read the input line and split it into words
            String line = value.toString();
            String[] words = line.trim().split("\\s+");

            // Skip lines with fewer than 2 words
            if (words.length < 2) 
                return;

            // Iterate through the words to generate stripes
            for (int i = 0; i < words.length - 1; i++) {
                // Clean up the words by removing non-alphabetic characters and extra apostrophes
                String currentWord = words[i].replaceAll("[^a-zA-Z']", "").replaceAll("^'+", "").replaceAll("'+$", "");
                String nextWord = words[i + 1].replaceAll("[^a-zA-Z']", "").replaceAll("^'+", "").replaceAll("'+$", "");

                // Skip empty words
                if (currentWord.isEmpty() || nextWord.isEmpty()) {
                    continue;
                }

                // Emit the current word as the key and a stripe containing the next word
                KEY.set(currentWord);
                STRIPE.clear();
                STRIPE.put(nextWord, 1);
                context.write(KEY, STRIPE);
            }
        }
    }

    /*
     * Reducer: Aggregates all stripes associated with each key and calculates relative frequencies.
     */
    private static class MyReducer extends Reducer<Text, HashMapStringIntWritable, PairOfStrings, FloatWritable> {

        // Reuse objects to save overhead of object creation.
        private final static HashMapStringIntWritable SUM_STRIPES = new HashMapStringIntWritable();
        private final static PairOfStrings BIGRAM = new PairOfStrings();
        private final static FloatWritable FREQ = new FloatWritable();

        @Override
        public void reduce(Text key, Iterable<HashMapStringIntWritable> stripes, Context context) throws IOException, InterruptedException {
            // Clear the aggregated stripe for the current key
            SUM_STRIPES.clear();

            // Aggregate all stripes for the current key
            for (HashMapStringIntWritable stripe : stripes) {
                for (Map.Entry<String, Integer> entry : stripe.entrySet()) {
                    String b = entry.getKey();
                    int count = entry.getValue();
                    SUM_STRIPES.increment(b, count);
                }
            }

            // Calculate the total count for the current key
            int sumTotal = 0;
            for (Map.Entry<String, Integer> entry : SUM_STRIPES.entrySet()) {
                sumTotal += entry.getValue();
            }

            // Skip if the total count is zero
            if (sumTotal == 0) {
                return;
            }

            // Emit the total count for the current word
            BIGRAM.set(key.toString(), "");
            FREQ.set(sumTotal);
            context.write(BIGRAM, FREQ);

            // Emit each bigram and its relative frequency
            for (Map.Entry<String, Integer> entry : SUM_STRIPES.entrySet()) {
                String b = entry.getKey();
                int count = entry.getValue();
                float frequency = (float) count / sumTotal;

                BIGRAM.set(key.toString(), b);
                FREQ.set(frequency);
                context.write(BIGRAM, FREQ);
            }
        }
    }

    /*
     * Combiner: Aggregates all stripes with the same key to reduce data transfer.
     */
    private static class MyCombiner extends Reducer<Text, HashMapStringIntWritable, Text, HashMapStringIntWritable> {

        // Reuse objects to save overhead of object creation.
        private final static HashMapStringIntWritable SUM_STRIPES = new HashMapStringIntWritable();

        @Override
        public void reduce(Text key, Iterable<HashMapStringIntWritable> stripes, Context context) throws IOException, InterruptedException {
            // Clear the aggregated stripe for the current key
            SUM_STRIPES.clear();

            // Aggregate all stripes for the current key
            for (HashMapStringIntWritable stripe : stripes) {
                for (Map.Entry<String, Integer> entry : stripe.entrySet()) {
                    String b = entry.getKey();
                    int count = entry.getValue();
                    SUM_STRIPES.increment(b, count);
                }
            }

            // Emit the aggregated stripe for the current key
            HashMapStringIntWritable outStripe = new HashMapStringIntWritable();
            outStripe.putAll(SUM_STRIPES);
            context.write(key, outStripe);
        }
    }

    /**
     * Creates an instance of this tool.
     */
    public BigramFrequencyStripes() {
    }

    private static final String INPUT = "input";
    private static final String OUTPUT = "output";
    private static final String NUM_REDUCERS = "numReducers";

    /**
     * Runs this tool.
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
        String outputPath = cmdline.getOptionValue(OUTPUT);
        int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

        LOG.info("Tool: " + BigramFrequencyStripes.class.getSimpleName());
        LOG.info(" - input path: " + inputPath);
        LOG.info(" - output path: " + outputPath);
        LOG.info(" - number of reducers: " + reduceTasks);

        // Create and configure a MapReduce job
        Configuration conf = getConf();
        Job job = Job.getInstance(conf);
        job.setJobName(BigramFrequencyStripes.class.getSimpleName());
        job.setJarByClass(BigramFrequencyStripes.class);

        job.setNumReduceTasks(reduceTasks);

        FileInputFormat.setInputPaths(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(HashMapStringIntWritable.class);
        job.setOutputKeyClass(PairOfStrings.class);
        job.setOutputValueClass(FloatWritable.class);

        /*
         * A MapReduce program consists of four components: a mapper, a reducer,
         * an optional combiner, and an optional partitioner.
         */
        job.setMapperClass(MyMapper.class);
        job.setCombinerClass(MyCombiner.class);
        job.setReducerClass(MyReducer.class);

        // Delete the output directory if it exists already.
        Path outputDir = new Path(outputPath);
        FileSystem.get(conf).delete(outputDir, true);

        // Time the program
        long startTime = System.currentTimeMillis();
        job.waitForCompletion(true);
        LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

        return 0;
    }

    /**
     * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
     */
    public static void main(String[] args) throws Exception {
        ToolRunner.run(new BigramFrequencyStripes(), args);
    }
}