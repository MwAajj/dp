package classifier.variants.basic;

import classifier.variants.Variant;
import classifier.structure.Structure;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class Knn implements Variant {
    private final Structure structure;
    private final int k;

    public Knn(Structure structure, int k) {
        this.structure = structure;
        this.k = k;
    }

    @Override
    public double[] distributionForInstance(Instance instance, int m_NumClasses) {
        double[] result = new double[m_NumClasses];
        Instances kNearestNeighbours = structure.findKNearestNeighbours(instance, k);
        double[] distances = structure.getDistances();
        double total = 0d;

        sortInstances(kNearestNeighbours, distances);

        for (int i = 0; i < kNearestNeighbours.numInstances(); i++) {
            Instance current = kNearestNeighbours.instance(i);
            result[(int) current.classValue()] += 1;
            total++;
        }
        for (int i = 0; i < result.length; i++) {
            result[i] = result[i] / total;
        }
        return result;
    }

    public void sortInstances(Instances neighbours, double[] distances) {
        sortNearestInstances(neighbours, distances);
    }

    public static void sortNearestInstances(Instances neighbours, double[] distances) {
        DistInst[] x = new DistInst[neighbours.size()];
        for (int i = 0; i < neighbours.size(); i++) {
            x[i] = new DistInst(neighbours.get(i), distances[i]);
        }
        Arrays.sort(x, Collections.reverseOrder());
        for (int i = 0; i < neighbours.size(); i++) {
            neighbours.set(i, x[i].getInstance());
        }
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
