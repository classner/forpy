// This is an adapted version of the excellent CTPL library
// (https://raw.githubusercontent.com/vit-vit/CTPL/master/ctpl_stl.h). The
// original copyright notice from Vitaliy Vitsentiy can be found below. There
// have been minor changes to the file for the use with forpy.

/*********************************************************
 *
 *  Copyright (C) 2014 by Vitaliy Vitsentiy
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *********************************************************/

#ifndef FORPY_UTIL_THREADING_CTPL_H_
#define FORPY_UTIL_THREADING_CTPL_H_

#include "../../global.h"

#include <atomic>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "../../types.h"
#include "../desk.h"

namespace forpy {
namespace threading {

typedef forpy::Desk *INFOT;

namespace detail {
template <typename T>
class Queue {
 public:
  bool push(T const &value) {
    std::unique_lock<std::mutex> lock(this->mutex);
    this->q.push(value);
    return true;
  }
  // deletes the retrieved element, do not use for non integral types
  bool pop(T &v) {
    std::unique_lock<std::mutex> lock(this->mutex);
    if (this->q.empty()) return false;
    v = this->q.front();
    this->q.pop();
    return true;
  }
  bool empty() {
    std::unique_lock<std::mutex> lock(this->mutex);
    return this->q.empty();
  }

 private:
  std::queue<T> q;
  std::mutex mutex;
};
}  // namespace detail

class thread_pool {
 public:
  thread_pool() { this->init(); }
  thread_pool(int nThreads) {
    this->init();
    this->resize(nThreads);
  }

  // the destructor waits for all the functions in the queue to be finished
  ~thread_pool() { this->stop(true); }

  // get the number of running threads in the pool
  int size() { return static_cast<int>(this->threads.size()); }

  // number of idle threads
  int n_idle() { return this->nWaiting; }
  std::thread &get_thread(int i) { return *this->threads[i]; }

  // change the number of threads in the pool
  // should be called from one thread, otherwise be careful to not interleave,
  // also with this->stop()
  // nThreads must be >= 0
  void resize(int nThreads) {
    if (!this->isStop && !this->isDone) {
      int oldNThreads = static_cast<int>(this->threads.size());
      if (oldNThreads <= nThreads) {  // if the number of threads is increased
        this->threads.resize(nThreads);
        this->flags.resize(nThreads);

        for (int i = oldNThreads; i < nThreads; ++i) {
          this->flags[i] = std::make_shared<std::atomic<bool>>(false);
          this->set_thread(i);
        }
      } else {  // the number of threads is decreased
        for (int i = oldNThreads - 1; i >= nThreads; --i) {
          *this->flags[i] = true;  // this thread will finish
          this->threads[i]->detach();
        }
        {
          // stop the detached threads that were waiting
          std::unique_lock<std::mutex> lock(this->mutex);
          this->cv.notify_all();
        }
        this->threads.resize(
            nThreads);  // safe to delete because the threads are detached
        this->flags.resize(nThreads);  // safe to delete because the threads
                                       // have copies of shared_ptr of the
                                       // flags, not originals
      }
    }
  }

  // empty the queue
  void clear_queue() {
    std::function<void(INFOT)> *_f;
    while (this->q.pop(_f)) delete _f;  // empty the queue
  }

  // pops a functional wrapper to the original function
  std::function<void(INFOT)> pop() {
    std::function<void(INFOT)> *_f = nullptr;
    this->q.pop(_f);
    std::unique_ptr<std::function<void(INFOT)>> func(
        _f);  // at return, delete the function even if an exception occurred
    std::function<void(INFOT)> f;
    if (_f) f = *_f;
    return f;
  }

  // wait for all computing threads to finish and stop all threads
  // may be called asynchronously to not pause the calling thread while waiting
  // if isWait == true, all the functions in the queue are run, otherwise the
  // queue is cleared without running the functions
  void stop(bool isWait = false) {
    if (!isWait) {
      if (this->isStop) return;
      this->isStop = true;
      for (int i = 0, n = this->size(); i < n; ++i) {
        *this->flags[i] = true;  // command the threads to stop
      }
      this->clear_queue();  // empty the queue
    } else {
      if (this->isDone || this->isStop) return;
      this->isDone = true;  // give the waiting threads a command to finish
    }
    {
      std::unique_lock<std::mutex> lock(this->mutex);
      this->cv.notify_all();  // stop all waiting threads
    }
    for (int i = 0; i < static_cast<int>(this->threads.size());
         ++i) {  // wait for the computing threads to finish
      if (this->threads[i]->joinable()) this->threads[i]->join();
    }
    // if there were no threads in the pool but some functors in the queue, the
    // functors are not deleted by the threads
    // therefore delete them here
    this->clear_queue();
    this->threads.clear();
    this->flags.clear();
  }

  /// For member functions (with parameters).
  template <typename C, typename F, typename M, typename... Rest>
  auto push_move(F &&f, C *c, M &&movable, Rest &&... rest)
      -> std::future<decltype((c->*f)(new forpy::Desk(0), movable, rest...))> {
    auto pck = std::make_shared<std::packaged_task<decltype(
        (c->*f)(new forpy::Desk(0), movable, rest...))(INFOT)>>(
        std::bind(std::forward<F>(f), std::forward<C *>(c),
                  std::placeholders::_1, std::move(movable),
                  std::forward<Rest>(rest)...));
    auto _f = new std::function<void(INFOT)>([pck](INFOT s) { (*pck)(s); });
    this->q.push(_f);
    std::unique_lock<std::mutex> lock(this->mutex);
    this->cv.notify_one();
    return pck->get_future();
  }

  /// For member functions (with parameters).
  template <typename C, typename F, typename... Rest>
  auto push(F &&f, C *c, Rest &&... rest)
      -> std::future<decltype((c->*f)(new forpy::Desk(0), rest...))> {
    auto pck = std::make_shared<std::packaged_task<decltype(
        (c->*f)(new forpy::Desk(0), rest...))(INFOT)>>(
        std::bind(std::forward<F>(f), std::forward<C *>(c),
                  std::placeholders::_1, std::forward<Rest>(rest)...));
    auto _f = new std::function<void(INFOT)>([pck](INFOT s) { (*pck)(s); });
    this->q.push(_f);
    std::unique_lock<std::mutex> lock(this->mutex);
    this->cv.notify_one();
    return pck->get_future();
  }

  /// For functions with parameters.
  template <typename F, typename... Rest>
  auto push(F &&f, Rest &&... rest)
      -> std::future<decltype(f(new forpy::Desk(0), rest...))> {
    auto pck = std::make_shared<
        std::packaged_task<decltype(f(new forpy::Desk(0), rest...))(INFOT)>>(
        std::bind(std::forward<F>(f), std::placeholders::_1,
                  std::forward<Rest>(rest)...));
    auto _f = new std::function<void(INFOT)>([pck](INFOT s) { (*pck)(s); });
    this->q.push(_f);
    std::unique_lock<std::mutex> lock(this->mutex);
    this->cv.notify_one();
    return pck->get_future();
  }

  /// For functions without parameters.
  template <typename F>
  auto push(F &&f) -> std::future<decltype(f(new forpy::Desk(0)))> {
    auto pck = std::make_shared<
        std::packaged_task<decltype(f(new forpy::Desk(0)))(INFOT)>>(
        std::forward<F>(f));
    auto _f = new std::function<void(INFOT)>([pck](INFOT s) { (*pck)(s); });
    this->q.push(_f);
    std::unique_lock<std::mutex> lock(this->mutex);
    this->cv.notify_one();
    return pck->get_future();
  }

  void init() {
    this->nWaiting = 0;
    this->isStop = false;
    this->isDone = false;
  }

 private:
  // deleted
  thread_pool(const thread_pool &) = delete;
  thread_pool(thread_pool &&) = delete;
  thread_pool &operator=(const thread_pool &) = delete;
  thread_pool &operator=(thread_pool &&) = delete;

  void set_thread(int i) {
    std::shared_ptr<std::atomic<bool>> flag(
        this->flags[i]);  // a copy of the shared ptr to the flag
    auto f = [this, i, flag /* a copy of the shared ptr to the flag */]() {
      VLOG(3) << "Starting up thread " << i << " (id "
              << std::this_thread::get_id() << ").";
      forpy::Desk d(i);
      std::atomic<bool> &_flag = *flag;
      std::function<void(INFOT)> *_f;
      bool isPop = this->q.pop(_f);
      while (true) {
        while (isPop) {  // if there is anything in the queue
          std::unique_ptr<std::function<void(INFOT)>> func(
              _f);  // at return, delete the function even if an exception
                    // occurred
          (*_f)(&d);
          d.reset();
          if (_flag)
            return;  // the thread is wanted to stop, return even if the queue
                     // is not empty yet
          else
            isPop = this->q.pop(_f);
        }
        // the queue is empty here, wait for the next command
        std::unique_lock<std::mutex> lock(this->mutex);
        ++this->nWaiting;
        this->cv.wait(lock, [this, &_f, &isPop, &_flag]() {
          isPop = this->q.pop(_f);
          return isPop || this->isDone || _flag;
        });
        --this->nWaiting;
        if (!isPop)
          return;  // if the queue is empty and this->isDone == true or *flag
                   // then return
      }
    };
    this->threads[i].reset(
        new std::thread(f));  // compiler may not support std::make_unique()
  }

  std::vector<std::unique_ptr<std::thread>> threads;
  std::vector<std::shared_ptr<std::atomic<bool>>> flags;
  detail::Queue<std::function<void(INFOT)> *> q;
  std::atomic<bool> isDone;
  std::atomic<bool> isStop;
  std::atomic<int> nWaiting;  // how many threads are waiting

  std::mutex mutex;
  std::condition_variable cv;
};
}  // namespace threading

class ThreadControl {
 private:
  inline ThreadControl() : ttp() {
    VLOG(1) << "Creating thread control (main thread id: "
            << std::this_thread::get_id() << ").";
  }
  DISALLOW_COPY_AND_ASSIGN(ThreadControl);
  std::unique_ptr<threading::thread_pool> ttp;

 public:
  inline static ThreadControl &getInstance() {
    static ThreadControl instance;
    return instance;
  }

  inline void set_num(size_t n) {
    if (n == 0) n = std::thread::hardware_concurrency();
    VLOG(1) << "Setting thread pool size to " << n << ".";
    if (ttp == nullptr) {
      VLOG(1) << "Initializing thread pool from scratch.";
      ttp = std::make_unique<threading::thread_pool>(n);
    } else {
      if (get_num() != n) {
        VLOG(1) << "Resizing thread pool.";
        ttp->resize(n);
      }
    }
  }

  inline size_t get_num() {
    if (ttp == nullptr)
      return 0;
    else
      return ttp->size();
  }

  inline size_t get_idle() {
    if (ttp == nullptr)
      return 0;
    else
      return ttp->n_idle();
  }

  /// For member functions (with parameters).
  template <typename C, typename F, typename M, typename... Rest>
  auto push_move(F &&f, C *c, M &&movable, Rest &&... rest)
      -> std::future<decltype((c->*f)(new forpy::Desk(0), movable, rest...))> {
    return ttp->push_move(f, c, movable, rest...);
  }

  template <typename C, typename F, typename... Rest>
  auto push(F &&f, C *c, Rest &&... rest)
      -> std::future<decltype((c->*f)(new forpy::Desk(0), rest...))> {
    return ttp->push(f, c, rest...);
  }

  template <typename F, typename... Rest>
  auto push(F &&f, Rest &&... rest)
      -> std::future<decltype(f(new Desk(0), rest...))> {
    return ttp->push(f, rest...);
  }

  template <typename F>
  auto push(F &&f) -> std::future<decltype(f(new Desk(0)))> {
    return ttp->push(f);
  }

  inline void stop(const bool &wait = false) {
    ttp->stop(wait);
    ttp->init();
  }

  inline ~ThreadControl() {
    VLOG(1) << "Destroying thread control...";
    ttp.reset();
    VLOG(1) << "Done.";
  }
};

}  // namespace forpy

#endif  // FORPY_UTIL_THREADING_CTPL_H_
