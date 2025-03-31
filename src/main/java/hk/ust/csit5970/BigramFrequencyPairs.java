package hk.ust.csit5970;

import java.io.IOException;
import java.util.Arrays;

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
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

/**
 * Compute the bigram count using the "pairs" approach.
 */
public class BigramFrequencyPairs extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(BigramFrequencyPairs.class);

    /*
     * Mapper: Processes input text and emits bigrams with a count of 1.
     */
    private static class MyMapper extends Mapper<LongWritable, Text, PairOfStrings, IntWritable> {

        // Reuse objects to save overhead of object creation.
        private static final IntWritable ONE = new IntWritable(1);
        private static final PairOfStrings BIGRAM = new PairOfStrings();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // Read the input line and split it into words
            String line = value.toString();
            String[] words = line.trim().split("\\s+");

            // Skip lines with fewer than 2 words
            if (words.length < 2) 
                return;

            // Iterate through the words to generate bigrams
            for (int i = 0; i < words.length - 1; i++) {
                // Clean up the words by removing non-alphabetic characters and extra apostrophes
                String word1 = words[i].replaceAll("[^a-zA-Z']", "").replaceAll("'+$", "").replaceAll("^'+", "");
                String word2 = words[i + 1].replaceAll("[^a-zA-Z']", "").replaceAll("'+$", "").replaceAll("^'+", "");

                // Emit the bigram and a special bigram with "*" for marginal counts
                if (!word1.isEmpty() && !word2.isEmpty()) {
                    BIGRAM.set(word1, word2);
                    context.write(BIGRAM, ONE);

                    BIGRAM.set(word1, "*");
                    context.write(BIGRAM, ONE);
                }
            }
        }
    }

    /*
     * Reducer: Calculates the relative frequency of each bigram.
     */
    private static class MyReducer extends Reducer<PairOfStrings, IntWritable, PairOfStrings, FloatWritable> {
        private String currentLeft = null; // Tracks the current left word
        private int currentSum = 0; // Tracks the total count for the current left word

        // Reuse objects to save overhead of object creation.
        private final static FloatWritable VALUE = new FloatWritable();

        @Override
        public void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            String left = key.getLeftElement();
            String right = key.getRightElement();

            // If the left word changes, reset the current sum
            if (!left.equals(currentLeft)) {
                currentLeft = left;
                currentSum = 0;
            }

            // Handle the special bigram with "*" to calculate the total count for the left word
            if (right.equals("*")) {
                int sum = 0;
                for (IntWritable val : values) {
                    sum += val.get();
                }
                currentSum = sum;

                // Emit the total count for the left word
                PairOfStrings outputKey = new PairOfStrings(left, "");
                VALUE.set(currentSum);
                context.write(outputKey, VALUE);
            } else {
                // Calculate the relative frequency for the bigram
                if (currentSum == 0) {
                    LOG.error("No sum found for left element: " + left);
                    return;
                }
                int count = 0;
                for (IntWritable val : values) {
                    count += val.get();
                }
                float freq = (float) count / currentSum;
                VALUE.set(freq);
                context.write(key, VALUE);
            }
        }
    }

    /*
     * Combiner: Combines intermediate results to reduce data transfer.
     */
    private static class MyCombiner extends Reducer<PairOfStrings, IntWritable, PairOfStrings, IntWritable> {
        private static final IntWritable SUM = new IntWritable();

        @Override
        public void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            // Sum up the counts for each bigram
            int sum = 0;
            for (IntWritable value : values) {
                sum += value.get();
            }
            SUM.set(sum);
            context.write(key, SUM);
        }
    }

    /*
     * Partitioner: Ensures that bigrams with the same left word are sent to the same reducer.
     */
    private static class MyPartitioner extends Partitioner<PairOfStrings, IntWritable> {
        @Override
        public int getPartition(PairOfStrings key, IntWritable value, int numReduceTasks) {
            // Partition based on the hash code of the left word
            return (key.getLeftElement().hashCode() & Integer.MAX_VALUE) % numReduceTasks;
        }
    }

    /**
     * Creates an instance of this tool.
     */
    public BigramFrequencyPairs() {
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

        LOG.info("Tool: " + BigramFrequencyPairs.class.getSimpleName());
        LOG.info(" - input path: " + inputPath);
        LOG.info(" - output path: " + outputPath);
        LOG.info(" - number of reducers: " + reduceTasks);

        // Create and configure a MapReduce job
        Configuration conf = getConf();
        Job job = Job.getInstance(conf);
        job.setJobName(BigramFrequencyPairs.class.getSimpleName());
        job.setJarByClass(BigramFrequencyPairs.class);

        job.setNumReduceTasks(reduceTasks);

        FileInputFormat.setInputPaths(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        job.setMapOutputKeyClass(PairOfStrings.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(PairOfStrings.class);
        job.setOutputValueClass(FloatWritable.class);

        /*
         * A MapReduce program consists of three components: a mapper, a reducer,
         * a combiner (which reduces the amount of shuffle data), and a partitioner.
         */
        job.setMapperClass(MyMapper.class);
        job.setCombinerClass(MyCombiner.class);
        job.setPartitionerClass(MyPartitioner.class);
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
        ToolRunner.run(new BigramFrequencyPairs(), args);
    }
}