package structure;

import weka.core.Instance;

public final class MathOperation {

    public static double euclidDistance(Instance old, Instance newInstance) {
        if (old.numAttributes() != newInstance.numAttributes()) {
            throw new RuntimeException("CalculateDistance: Incompatible size of instances");
        }
        int classIndex = 0;
        try {
            classIndex = old.classIndex();
        } catch (Exception e) {
            throw new RuntimeException("Instance doesn't have class index");
        }
        double sum = 0;
        for (int i = 0; i < old.numAttributes(); i++) {
            if(i == classIndex) continue; // don't calculate distance for class index
            sum += Math.pow((old.value(i) - newInstance.value(i)), 2d);
        }
        double square = Math.sqrt(sum);
        return square;
    }
}
