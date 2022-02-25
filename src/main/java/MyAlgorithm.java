import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import structure.MathOperation;
import structure.Tree;
import structure.ballTree.BallTree;
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
    private boolean mk_knn = true;
    private boolean mk_HmdKnn = false;
    private boolean mk_FKnn = false;
    private int m_NumClasses = 0;
    private boolean mk_variance = false;
    private int k = 1;
    private int m;
    private int r;
    private Tree tree;

    public MyAlgorithm(int k) {
        this.k = k;
    }

    @Override
    public void buildClassifier(Instances data) {
        checkData(data);
        m_NumClasses = data.numClasses();
        if (mk_isKdTree)
            tree = new KdTree(mk_variance);
        else tree = new BallTree(k);
        tree.buildTree(data);
    }

    public void help(Instance instance) {
        Instances kNearestNeighbours = tree.findKNearestNeighbours(instance, k);
        for (Instance instance1 : kNearestNeighbours) {
            System.out.println(instance1);
        }
    }

    public double noWeight(Instance instance) {
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

    private double fuzzyKNN(Instance instance) {
        Instances instances = tree.findKNearestNeighbours(instance, k);
        double[] distances = tree.getDistances();
        double endClass = 0d, max = -1d;
        Map<Double, Double> info = new HashMap<>();
        for (int i = 0; i < m_NumClasses; i++) {
            double prob = MathOperation.fuzzyDistance(instances, distances, i, m);
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


    private double hmdKNN(Instance instance) {
        if (r > k)
            throw new RuntimeException("r parameter must be smaller then k");
        Instances instances = tree.findKNearestNeighbours(instance, k);
        double min = Double.MAX_VALUE, endClass = 0d;
        Map<Double, Double> info = new HashMap<>();
        for (int i = 0; i < m_NumClasses; i++) {
            double prob = MathOperation.newHarmonicDistance(instances, instance, k, r, m_NumClasses);
            if (min > prob) {
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

    //return end class of new instance
    @Override
    public double classifyInstance(Instance instance) {
        double endClass = 0d;
        if (mk_knn) {
            endClass = noWeight(instance);
        } else if (mk_FKnn) {
            endClass = fuzzyKNN(instance);
        } else if (mk_HmdKnn) {
            endClass = hmdKNN(instance);
        }
        return endClass;
    }

    //return probabilities for each class of new instance
    @Override
    public double[] distributionForInstance(Instance instance) {
        double[] result = new double[m_NumClasses];
        Instances kNearestNeighbours = tree.findKNearestNeighbours(instance, k);
        double[] distances = tree.getDistances();
        if(mk_FKnn) {
            for (int i = 0; i < m_NumClasses; i++) {
                double prob = MathOperation.fuzzyDistance(kNearestNeighbours, distances, i, m);
                result[i] = prob;
            }
            return result;
        } else if (mk_HmdKnn) {
            System.out.println("TODO");
        } else {
            double weight = 1, total = 0d;
            for (int i = 0; i < kNearestNeighbours.numInstances(); i++) {
                Instance current = kNearestNeighbours.instance(i);
                if(mk_knn) {
                    result[(int)current.classValue()] += 1;
                }
                else if (mk_FKnn) {
                    result[(int)current.classValue()] += weight;
                }
                total += weight;
            }
            for (int i = 0; i < result.length; i++) {
                result[i] = result[i] / total;
            }

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
            mk_HmdKnn = true;
            mk_knn = false;
            int r = Integer.parseInt(harmonicString);
            if (r > k) {
                System.err.println("For harmonic distance r {" + r + "} value shouldn't be bigger then k{" + k + "}");
                setR(k);
            } else setR(r);
        } else if (fuzzyString.length() != 0) {
            mk_FKnn = true;
            mk_knn = false;
            setM(Integer.parseInt(fuzzyString));
        }
        if(Utils.getFlag('V', options))
            mk_variance = true;
    }

    @Override
    public String[] getOptions() {
        Vector<String> options = new Vector<>();
        Collections.addAll(options, super.getOptions());
        options.add("-K");
        options.add("" + getK());
        if (mk_FKnn) {
            options.add("-F");
        } else if (mk_HmdKnn) {
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

    private void checkData(Instances data) {
        try {
            getCapabilities().testWithFail(data);
        } catch (Exception e) {
            throw new RuntimeException("Test failed: " + e);
        }
        if (data.size() < k) {
            throw new RuntimeException("K {" + k + "} is bigger then size of data{"
                    + data.size() + "}");
        }
    }
}
