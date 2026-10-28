/**
 * Jobs for a module worker, as promises.
 *
 * Each job carries an id and settles when the worker answers it with
 * { id, ok, result | error }. A worker whose script fails to load, that
 * crashes, or whose reply cannot be read fires an error instead of answering;
 * with no listener for that, the job waited forever and its spinner never
 * stopped. Here such a worker rejects every job it holds and is dropped, and
 * the next job starts a fresh one.
 *
 * cancel() terminates the worker and rejects its jobs with "cancelled": the
 * only way to stop a long job, such as a deadline map no one wants any more.
 *
 * @param {() => Worker} create  makes the worker
 * @param {{failure?: string}} opts  failure: the message a failed worker's jobs
 *                                   reject with
 */
export function workerClient(create, { failure = "A background task stopped. Reload the page and try again." } = {}) {
  let worker = null;
  const pending = new Map();

  const drop = (error) => {
    worker?.terminate();
    worker = null;
    const jobs = [...pending.values()];
    pending.clear();
    jobs.forEach((job) => job.reject(error));
  };

  const start = () => {
    const w = create();
    w.addEventListener("message", (event) => {
      const job = pending.get(event.data?.id);
      if (!job) return;
      pending.delete(event.data.id);
      if (event.data.ok) job.resolve(event.data.result);
      else job.reject(new Error(event.data.error || failure));
    });
    const failed = (event) => {
      event?.preventDefault?.();
      if (worker === w) drop(new Error(failure));
    };
    w.addEventListener("error", failed);
    w.addEventListener("messageerror", failed);
    return w;
  };

  return {
    run(message, transfers = []) {
      return new Promise((resolve, reject) => {
        const id = Math.random().toString(36).slice(2);
        try {
          worker ||= start();
          pending.set(id, { resolve, reject });
          worker.postMessage({ ...message, id }, transfers);
        } catch (error) {
          pending.delete(id);
          reject(error);
        }
      });
    },
    get busy() {
      return pending.size > 0;
    },
    cancel() {
      if (worker || pending.size) drop(new Error("cancelled"));
    },
  };
}

export const isCancelled = (error) => error?.message === "cancelled";
