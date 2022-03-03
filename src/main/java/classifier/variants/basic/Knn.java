package classifier.variants.basic;

import classifier.variants.Variant;
import structure.Tree;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;

public class Knn implements Variant {
    private Tree tree;
    private int k;

    public Knn(Tree tree, int k) {
        this.tree = tree;
        this.k = k;
    }

    @Override
    public double[] distributionForInstance(Instance instance, int m_NumClasses) {
        double[] result = new double[m_NumClasses];
        Instances kNearestNeighbours = tree.findKNearestNeighbours(instance, k);
        double weight = 1, total = 0d;
        for (int i = 0; i < kNearestNeighbours.numInstances(); i++) {
            Instance current = kNearestNeighbours.instance(i);
            result[(int) current.classValue()] += 1;
            total += weight;
        }
        for (int i = 0; i < result.length; i++) {
            result[i] = result[i] / total;
        }
        return result;
    }

    @Override
    public double classifyInstance(Instance instance, int m_NumClasses) {
        Instances instances = tree.findKNearestNeighbours(instance, k);
        Map<Double, Integer> occurrences = getOccurrences(instances);
        int max = Integer.MIN_VALUE;
        double endClass = -1d;

        for (Map.Entry<Double, Integer> pair : occurrences.entrySet()) {
            if (max < pair.getValue()) {
                max = pair.getValue();
                endClass = pair.getKey();
            }
        }
        return endClass;
    }

    @Override
    public String getOption() {
        return null;
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
}
