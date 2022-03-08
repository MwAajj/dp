package classifier.variants.basic;

import classifier.variants.Variant;
import structure.Structure;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;

public class Knn implements Variant {
    private Structure structure;
    private int k;

    public Knn(Structure structure, int k) {
        this.structure = structure;
        this.k = k;
    }

    @Override
    public double[] distributionForInstance(Instance instance, int m_NumClasses) {
        double[] result = new double[m_NumClasses];
        Instances kNearestNeighbours = structure.findKNearestNeighbours(instance, k);
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
        Instances instances = structure.findKNearestNeighbours(instance, k);
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
        return "";
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
