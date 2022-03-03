package classifier.variants;

import lombok.Getter;
import lombok.Setter;
import structure.MathOperation;
import structure.Tree;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;

@Getter
@Setter
public class FuzzyKnn implements Variant {
    Map<Double, Double> info = new HashMap<>();
    private Tree tree;
    private int k;
    private int m;

    public FuzzyKnn(Tree tree, int k, int m) {
        this.tree = tree;
        this.k = k;
        this.m = m;
    }

    @Override
    public double[] distributionForInstance(Instance instance, int m_NumClasses) {
        double[] result = new double[m_NumClasses];
        Instances kNearestNeighbours = tree.findKNearestNeighbours(instance, k);
        double[] distances = tree.getDistances();
        for (int i = 0; i < m_NumClasses; i++) {
            double prob = MathOperation.fuzzyDistance(kNearestNeighbours, distances, i, m);
            result[i] = prob;
        }
        return result;
    }

    @Override
    public double classifyInstance(Instance instance, int m_NumClasses) {
        Instances instances = tree.findKNearestNeighbours(instance, k);
        double[] distances = tree.getDistances();
        double endClass = 0d, max = -1d;

        for (int i = 0; i < m_NumClasses; i++) {
            double prob = MathOperation.fuzzyDistance(instances, distances, i, m);
            if (max < prob) {
                max = prob;
                endClass = i;
            }
            info.put((double) i, prob);
        }
        return endClass;
    }

    @Override
    public String getOption() {
        return null;
    }

    private void printInfo() {
        System.out.println("\n--------------------F-KNN------------");
        System.out.println(info);
        System.out.println("-------------------------------------");
    }
}
