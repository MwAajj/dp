import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import structure.MathOperation;
import structure.Tree;
import structure.balltree.BallTree;
import structure.kdtree.KdTree;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.*;

import java.util.*;

@Getter
@Setter
@NoArgsConstructor
public class MyAlgorithm extends AbstractClassifier implements Classifier, OptionHandler {

    private boolean mk_isKdTree = false;
    private boolean mk_isBallTree = false;
    private boolean mk_noWeight = false;
    private boolean mk_harmonicWeight = false;
    private boolean mk_fuzzyWeight = false;
    private int m_NumClasses = 0;
    private int k;
    private int m;
    private int r;
    private Tree tree;

    public MyAlgorithm(int k) {
        this.k = k;
    }


    @Override
    public void buildClassifier(Instances data) {
        m_NumClasses = data.numClasses();
        if (data.size() < k) {
            System.err.println("K {" + k + "} is bigger then size of data{"
                    + data.size() + "}. You have been warned!!!");
        }
        if (mk_isBallTree)
            tree = new KdTree();
        else if (mk_isKdTree)
            tree = new KdTree();
        tree.buildTree(data);
    }

    private double noWeight(Instance instance) {
        Instances instances = tree.findKNearestNeighbours(instance, k);
        Map<Double, Integer> occurrences = getOccurrences(instances);
        int max = Integer.MIN_VALUE;
        double endClass = 0d;
        for (Map.Entry<Double, Integer> pair : occurrences.entrySet()) {
            if (max < pair.getValue()) {
                max = pair.getValue();
                endClass = pair.getKey();
            }
        }
        return endClass;
    }

    private Map<Double, Integer> getOccurrences(Instances instances) {
        Map<Double, Integer> occurrences = new HashMap<>();
        for (Instance instance : instances) {
            double val;
            try {
                val = instance.classValue();
            } catch (Exception E) {
                throw new Error("Data has no class attribute!");
            }
            Integer count = occurrences.get(val);
            occurrences.put(val, count != null ? count + 1 : 1);
        }
        return occurrences;
    }

    private double mk_fuzzyWeight(Instance instance) {
        Instances instances = tree.findKNearestNeighbours(instance, k);
        double endClass = 0d, max = -1d;
        Map<Double, Double> info = new HashMap<>();
        for (int i = 0; i < m_NumClasses; i++) {
            double prob = MathOperation.fuzzyDistance(instances, instance, i, m);
            if (max < prob) {
                max = prob;
                endClass = i;
            }
            info.put((double) i, prob);
        }
        System.out.println("\n--------------------F-KNN------------");
        System.out.println(info);
        System.out.println("-------------------------------------");
        return endClass;
    }

    private double mk_harmonicWeight(Instance instance) {
        Instances instances = tree.findKNearestNeighbours(instance, k);
        /*double[] meanDistances = new double[m_NumClasses];
        Map<Double, Double> info = new HashMap<>();
        double endClass = -1d;
        double min = Double.MAX_VALUE;
        for (int i = 0; i < m_NumClasses; i++) {
            double prob = MathOperation.hmDistance(instances, instance, i, this.k);
            if(min > prob) {
                min = prob;
                endClass = i;
            }
            info.put((double) i, prob);
        }*/

        double[] meanDistances = new double[m_NumClasses];
        int index = 0;
        for (int i = 0; i < m_NumClasses; i++) {
            double prob = MathOperation.meanDistances(instances, instance, i, this.r);
            meanDistances[index] = prob;
            index++;
        }
        double[] harmonicMeanDistances = new double[m_NumClasses];
        index = 0;
        for (int i = 0; i < m_NumClasses; i++) {
            double prob = MathOperation.harmonicDistance(instances, meanDistances, i, this.r);
            harmonicMeanDistances[index] = prob;
            index++;
        }
        double min = Double.MAX_VALUE;
        double endClass = 0d;
        index = 0;
        Map<Double, Double> info = new HashMap<>();
        for (int i = 0; i < m_NumClasses; i++) {
            double prob = MathOperation.newHarmonicDistance(harmonicMeanDistances[i], this.k, m_NumClasses);
            index++;
            if(min > prob) {
                min = prob;
                endClass = i;
            }
            info.put((double) i, prob);
        }
        System.out.println("\n--------------------HMD-KNN------------");
        System.out.println(info);
        System.out.println("-------------------------------------");
        return endClass;
    }


    @Override
    public double classifyInstance(Instance instance) {
        double endClass = 0d;
        if (mk_noWeight) {
            endClass = noWeight(instance);
        } else if (mk_fuzzyWeight) {
            endClass = mk_fuzzyWeight(instance);
        } else if (mk_harmonicWeight) {
            endClass = mk_harmonicWeight(instance);
        }
        return endClass;
    }


    @Override
    public double[] distributionForInstance(Instance instance) {
        double[] result = new double[m_NumClasses];
        Instances kNearestNeighbours = tree.findKNearestNeighbours(instance, k);
        Map<Double, Integer> occurrences = getOccurrences(kNearestNeighbours);
        for (Map.Entry<Double, Integer> pair : occurrences.entrySet()) {
            int index = (int) pair.getKey().doubleValue();
            result[index] = pair.getValue() / (double) k;
        }
        return result;
    }

    @Override
    public Enumeration<Option> listOptions() {
        return null;
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String knnString = Utils.getOption('K', options);
        if (knnString.length() != 0) {
            setK(Integer.parseInt(knnString));
        }
        if (Utils.getFlag('B', options))
            mk_isBallTree = true;
        else
            mk_isKdTree = true;
        String harmonicString = Utils.getOption('H', options);
        String fuzzyString = Utils.getOption('F', options);
        if (harmonicString.length() != 0) {
            mk_harmonicWeight = true;
            int r = Integer.parseInt(harmonicString);
            if (r > k) {
                System.err.println("For harmonic distance r {" + r + "} value shouldn't be bigger then k{" + k + "}");
                setR(k);
            } else setR(r);
        } else if (fuzzyString.length() != 0) {
            mk_fuzzyWeight = true;
            setM(Integer.parseInt(fuzzyString));
        } else mk_noWeight = true;
    }

    @Override
    public String[] getOptions() {
        Vector<String> options = new Vector<>();
        Collections.addAll(options, super.getOptions());
        options.add("-K");
        options.add("" + getK());
        if (mk_fuzzyWeight) {
            options.add("-F");
        } else if (mk_harmonicWeight) {
            options.add("-H");
        }
        return options.toArray(new String[0]);
    }


    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.DATE_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.DATE_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);
        return result;
    }
}
